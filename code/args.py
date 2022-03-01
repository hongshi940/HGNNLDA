import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/lncRNA_disease_test/',
                        help='path to data')
    parser.add_argument('--model_path', type=str, default='../model_save/',
                        help='path to save model')
    parser.add_argument('--D_n', type=int, default=412,
                        help='number of disease node')
    parser.add_argument('--L_n', type=int, default=240,
                        help='number of lncRNA node')
    parser.add_argument('--M_n', type=int, default=495,
                        help='number of miRNA node')
    parser.add_argument('--in_f_d', type=int, default=128,
                        help='input feature dimension')
    parser.add_argument('--embed_d', type=int, default=128,
                        help='embedding dimension')
    parser.add_argument('--lr', type=int, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_s', type=int, default=20000,
                        help='batch size')
    parser.add_argument('--mini_batch_s', type=int, default=200,
                        help='mini batch size')
    parser.add_argument('--train_iter_n', type=int, default=200,
                        help='max number of training iteration')
    parser.add_argument('--walk_n', type=int, default=20,
                        help='number of walk per root node')
    parser.add_argument('--walk_L', type=int, default=30,
                        help='length of each walk')
    parser.add_argument('--window', type=int, default=5,
                        help='window size for relation extration')
    parser.add_argument("--random_seed", default=10, type=int)
    parser.add_argument('--train_test_label', type=int, default=0,
                        help='train/test label: 0 - train, 1 - test, 2 - code test/generate negative ids for evaluation')
    parser.add_argument('--save_model_freq', type=float, default=2,
                        help='number of iterations to save model')
    parser.add_argument("--cuda", default=0, type=int)
    parser.add_argument("--checkpoint", default='', type=str)

    args = parser.parse_args()

    return args
