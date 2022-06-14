"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
import logging
import sys
import time

import patch_config
from load_data import *

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        #self.log_file_config = os.path.join(saved_patches/)

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.patch_applier = PatchApplier(patch_alpha=self.config.patch_alpha).cuda()
        self.patch_transformer = PatchTransformer(target_size_frac=self.config.target_size_frac).cuda()
        # AVE Edit below:
        self.prob_extractor = MaxProbExtractor(self.config.cls_id, self.config.num_cls, self.config).cuda()
        # Original below
        # self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        # create logs
        # self.writer_config = self.init_tensorboard(mode + '_config')
        # self.writer_config.add_text('mode', mode)
        self.writer = self.init_tensorboard(mode)
        

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        # make output dir
        patch_dir = os.path.join(self.config.src_dir, 'saved_patches', self.config.patch_name)
        os.makedirs(patch_dir, exist_ok=True)
        log_file = os.path.join(patch_dir, self.config.patch_name + '_log.txt')
        
        # python logging
        ###############################################################################
        # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
        # set up logging to file - see previous section for more details
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=log_file,
            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        ###############################################################################
        
        # print config values to log
        logging.info("patch_dir: {}".format(patch_dir))
        logging.info("config_dict:")
        config_dict = {key:value for key, value in self.config.__dict__.items() if not key.startswith('__') and not callable(key)}
        for k,v in config_dict.items():
            logging.info("key={}\t val={}".format(k, v))
        # logging.info("{x}".format(x=self.config_dict))
        # self.writer_config.add_text('config', self.config)
        
        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        # AVE Edit below
        n_epochs = self.config.n_epochs
        max_lab = self.config.max_lab
        # Original below
        # n_epochs = 10000
        # max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        if self.config.patchfile == 'gray':
            adv_patch_cpu = self.generate_patch("gray")
        elif self.config.patchfile == 'random':
            adv_patch_cpu = self.generate_patch("random")
        else:
            adv_patch_cpu = self.read_image(self.config.patchfile)
        adv_patch_cpu.requires_grad_(True)
        
        # AVE Edit Below
        train_loader = torch.utils.data.DataLoader(CamoDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                               shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        for epoch in range(n_epochs):
            out_patch_path = os.path.join(patch_dir, self.config.patch_name + '_epoch' + str(epoch) + '.jpg')
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    if (i_batch % 100) == 0:
                        logging.info('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                    img = p_img_batch[1, :, :,]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    #img.show()

                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps*self.config.nps_mult
                    tv_loss = tv*self.config.tv_mult
                    # nps_loss = nps*0.01
                    # tv_loss = tv*2.5
                    det_loss = torch.mean(max_prob)
                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                logging.info('  EPOCH NR: {}'.format(epoch)),
                logging.info('EPOCH LOSS: {}'.format(ep_loss))
                logging.info('  DET LOSS: {}'.format(ep_det_loss))
                logging.info('  NPS LOSS: {}'.format(ep_nps_loss))
                logging.info('   TV LOSS: {}'.format(ep_tv_loss))
                logging.info('EPOCH TIME: {}'.format(et1-et0))
                
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                # plt.imshow(im)
                # plt.show()
                im.save(out_patch_path)
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)

    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


