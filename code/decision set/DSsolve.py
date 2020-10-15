import sys
from SplitMaxSATDSsolve import SplitMaxSATDSSolve
from WholeMaxSATDSsolve import WholeMaxSATDSSolve

from WholeIterativeSATDSsolve import WholeIterativeSATDSSolve
from SplitIterativeSATDSsolve import SplitIterativeSATDSSolve

from SparseMaxSATDSsolve import SparseMaxSATDSSolve
from SepSparseMaxSATDSsolve import SepSparseMaxSATDSSolve
import getopt
import os


# filename = "test.csv"
# filename = "test2.csv"

# filename = "appendicitis.csv"
#filename = "backache.csv"
#filename = "balance.csv"
# filename = "biomed.csv"
#filename = "blood-transfusion.csv"
#filename = "breast-cancer.csv" #overlap
#filename = "bupa.csv"
#filename = "car.csv"
#filename = "cancer.csv"
#filename = "cleveland-nominal.csv"
#filename = "cloud.csv"
# filename = "colic.csv"
# filename = "contraceptive.csv"
#filename = "corral.csv"
#filename = "dataset_1.csv"
# filename = "dataset_2.csv"
# filename = "dataset_3.csv"
# filename = "dataset_4.csv"
# filename = "dermatology.csv"
# filename = "features10-train.csv" overlap
#filename = "flags.csv"
# filename = "haberman.csv" #overlap
#filename = "hayes-roth.csv"
# filename = "heart-h.csv"
#filename = "hepatitis.csv"
#filename = "house-votes-84.csv"
#filename = "iris.csv"
#filename = "irish.csv"
# filename = "liver-disorder.csv"
#filename = "lupus.csv"
#filename = "lymphography.csv"
# filename = "minizinc_class.csv"
#filename = "molecular-biology_promoters.csv"
#filename = "mouse-un.csv"
#filename = "mux6.csv"
# filename = "new-thyroid.csv"
#filename = "postoperative-patient-data.csv"
#filename = "promoters.csv"
# filename = "soybean.csv"
#filename = "spect.csv" #inconsistent
#filename = "test.csv"
#filename = "titanic.csv" #overlap
#filename = "uci_mammo_data.csv" # overlap
filename = "weather.csv"
#filename = "zoo.csv"


def parse_options():
    """
        Parses command-line options.
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'h:l:m:n:w:',
                                   ['help',
                                    'lambda',
                                    'model',
                                    'nodes',
                                    'misweight'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize() + '\n')
        usage()
        sys.exit(1)

    model = None
    N = 10
    lam = 0.005
    mis_weight = 1
    train_filename = args[0]
    try:
        test_filename = args[1]
    except:
        if 'train' in train_filename:
            newName = train_filename
            test_filename = newName.replace("train", "test")
        else:
            test_filename = train_filename

    for opt, arg in opts:
        if opt in ('-m', '--model'):
            model = str(arg.strip())
        elif opt in ('-n', '--nodes'):
            N = int(arg)
        elif opt in ('-l', '--lambda'):
            try:
                lam = float(arg)
            except:
                pass
        elif opt in ('-w', '--weight', '--misweight'):
            mis_weight = int(arg)
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return model, N, lam, mis_weight, train_filename, test_filename

#
#==============================================================================
def usage():
    """
        Prints help message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] file')
    print('Options:')
    print('        -m, --model            Model')
    print('        -n, --nodes            Number of nodes')
    print('        -l, --lambda           Value of lambda')
    print('        -w, --misweight        Weight of misclassification')
    print('        -h, --help             Print this usage message')



if __name__ == "__main__":

    no_ins = 1  # Testing time
    N = 1  # Number of nodes
    lam = 0.005
    mis_weight = 1
    model = '1'
    train_filename = filename
    test_filename = filename
    #model, N, lam, mis_weight, train_filename, test_filename = parse_options()
    if model == '1':
        WholeIterativeSATDSSolve(train_filename, no_ins, N)
    elif model == '2':
        SplitIterativeSATDSSolve(train_filename, no_ins, N)
    elif model == '3':
        WholeMaxSATDSSolve(train_filename, no_ins, N)
    elif model == '4':
        SplitMaxSATDSSolve(train_filename, no_ins, N)
    elif model == '5':
        SparseMaxSATDSSolve(train_filename, test_filename, no_ins, N, lam, mis_weight)
    elif model == '6':
        SepSparseMaxSATDSSolve(train_filename, test_filename, no_ins, N, lam, mis_weight)
    else:
        assert 'Invalid input'
