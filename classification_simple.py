from classification_Deepddi import Tester
from dataset import Dataset
import argparse
import time
def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=100, type=int, help="number of epochs")
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="DeepDDI", type=str, help="DeepDDI")
    parser.add_argument('-emb_dim', default=200, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-batch_size', default=512, type=int, help="batch size")
    parser.add_argument('-save_each', default=100, type=int, help="validate every k epochs")
    parser.add_argument('-discriminator_range', default=20, type=int, help="adiscriminator range")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    dataset = Dataset(args.dataset)

    print("~~~~ Testing on the 100 epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + 'simple' + "/" + '100' + ".chkpnt"
    tester = Tester(dataset, best_model_path)
    tester.test()

    print("~~~~ Testing on the 200 epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + 'simple' + "/" + '200' + ".chkpnt"
    tester = Tester(dataset, best_model_path)
    tester.test()

    print("~~~~ Testing on the 300 epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + 'simple' + "/" + '300' + ".chkpnt"
    tester = Tester(dataset, best_model_path)
    tester.test()

    print("~~~~ Testing on the 400 epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + 'simple' + "/" + '400' + ".chkpnt"
    tester = Tester(dataset, best_model_path)
    tester.test()

    print("~~~~ Testing on the 500 epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + 'simple' + "/" + '500' + ".chkpnt"
    tester = Tester(dataset, best_model_path)
    tester.test()

    print("~~~~ Testing on the 600 epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + 'simple' + "/" + '600' + ".chkpnt"
    tester = Tester(dataset, best_model_path)
    tester.test()

    print("~~~~ Testing on the 700 epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + 'simple' + "/" + '700' + ".chkpnt"
    tester = Tester(dataset, best_model_path)
    tester.test()

    print("~~~~ Testing on the 800 epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + 'simple' + "/" + '800' + ".chkpnt"
    tester = Tester(dataset, best_model_path)
    tester.test()

    print("~~~~ Testing on the 900 epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + 'simple' + "/" + '900' + ".chkpnt"
    tester = Tester(dataset, best_model_path)
    tester.test()

    print("~~~~ Testing on the 1000 epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + 'simple' + "/" + '1000' + ".chkpnt"
    tester = Tester(dataset, best_model_path)
    tester.test()