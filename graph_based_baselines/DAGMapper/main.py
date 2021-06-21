from arguments import get_parser, update_dir
from main_train import run_train
from main_val import run_val
from main_env import Environment
import time

def main():
    parser = get_parser()
    args = parser.parse_args()
    update_dir(args)
    if args.pretrain:
        args.epochs = 10
    print('===================== Parameters =======================')
    print('Device: {}'.format(args.device))
    print('Batch size: {}'.format(args.batch_size))
    print('Epoch number: {}'.format(args.epochs))
    print('Pretrain: {}'.format(args.pretrain))
    print('Learning rate: {} || Decay rate: {}'.format(args.lr_rate,args.weight_decay))
    print('===================== Starting modified DAGMapper =======================')
    env = Environment(args)
    if args.test:
        time_start = time.time()
        env.network.val_mode()
        run_val(env,0)
        print('Testing time usage: {}h'.format(round((time.time()-time_start)/3600,3)))
    else:
        for epoch in range(args.epochs):
            env.epoch_counter += 1
            for iCurb_image_index, data in enumerate(env.network.dataloader_train):
            # load a single one tiff image and run training on this image
                # validation mode
                if (iCurb_image_index%1000)==0 and iCurb_image_index:
                    env.network.val_mode()
                    f1 = run_val(env,iCurb_image_index)
                    if f1 > env.network.best_f1:
                        env.network.best_f1 = f1
                        env.network.save_checkpoints(iCurb_image_index)
                # training mode
                env.network.train_mode()
                run_train(env,data,iCurb_image_index)
                
if __name__ == "__main__":
    main()