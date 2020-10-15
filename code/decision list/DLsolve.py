import sys
from solve import Solve
import getopt
import os

def parse_options():
    """
        Parses command-line options.
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'a:h:l:m:n:o:',
                                   ['help',
                                    'lam',
                                    'lambda',
                                    'maxsat',
                                    'mxsat',
                                    'ml',
                                    'model',
                                    'node',
                                    'order',
                                    'sep',
                                    'separate',
                                    'spa',
                                    'sparse'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize() + '\n')
        usage()
        sys.exit(1)
    '''
    print(opts)
    print(args)
    '''

    train_filename = args[0]
    try:
        test_filename = args[1]
    except:
        test_filename = train_filename
        '''
        if 'train' in train_filename:
            newName = train_filename[:]
            test_filename = newName.replace("train", "test")
        else:

            test_filename = train_filename
        '''
    lam = 0.005
    maxsat = True # If a maxsat model
    ml = 'list'  # 'list, 'hybrid'
    N = 1
    sep_order = 'all' #'all', 'maj','accuy', 'cost' in separated models
    asc_order = 'desc'
    sep =  False # If a separated model
    sparse = False # If a sparse model

    for opt, arg in opts:
        if opt in ('-a'):
            asc_order = str(arg)
            assert asc_order in ('asc', 'desc', 'both'), 'option must be in (asc, desc, both)'
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-l', '--lam', '--lambda'):
            lam = float(arg)
        elif opt in ('--maxsat', '--mxsat'):
            maxsat = True
        elif opt in ('-m', '--model'):
            ml = str(arg)
            assert ml in ('list', 'hybrid'), 'option must be in (list, hybrid)'
        elif opt in ('-n', '--node'):
            N = int(arg)
        elif opt in ('-o', '--order'):
            sep_order = str(arg)
            assert sep_order in ('all', 'maj', 'accuy', 'cost'), 'option must be in (all, maj, accuy, cost)'
        elif opt in ('--sep', '--separate'):
            sep = True
        elif opt in ('--spa', '--sparse'):
            sparse = True

        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)


    return train_filename, test_filename, N, lam, sep_order, asc_order, maxsat, sep, ml, sparse
#
#==============================================================================
def usage():
    """
        Prints help message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] file')
    print('Options:')
    print('        -a                               The order)')
    print('                                         \'asc\': smallest/lowest first')
    print('                                         \'desc\': larggest/highest first')
    print('        -h, --help                       Print this usage message')
    print('        -l, --lam, --lambda==<float>     Value of lambda')
    print('        --maxsat, --mxsat                It is a MaxSAT model')
    print('        -m, --model==<string>            The model is a list, hybrid model')
    print('        -n, --nodes==<int>               The initial upper bound')
    print('        -o, --order==<string>            The class order in a separated model')
    print('                                         \'all\': output all possible orders')
    print('                                         \'maj\': output the order based on the majority of class')
    print('                                         \'accuy\': output the order based on the accuracy in each class')
    print('        --sep, --separate                It is a separated model')
    print('        --spa, --sparse                  It is a sparse model')


if __name__ == "__main__":


        '''
        file = open('solved' + str(i) +'.txt')
        train_filenames = file.read().split()
        file.close()
        #train_filenames = ['r2/appendicitis_train4.csv.gz']

        benchmark = []
        benchmark.append(
            ['Filename', 'Items', 'Features', 'Class', 'Solved', 'Nodes', 'Rule', 'Literal', 'Train_accuracy',
             'Test_accuracy', 'Runtime'])

        for train_filename in train_filenames:
            newName = train_filename[:]
            test_filename = newName.replace("train", "test")
            N = 1
            lam = 0.05
            sep_order = 'maj'
            asc_order = 'desc'
            maxsat = True
            sep = True
            ml = 'list'
            sparse = True

        '''
        
        train_filename, test_filename, N, lam, sep_order, asc_order, maxsat, sep, ml, sparse = parse_options()

        print('train_filename:', train_filename)
        #print('test_filename:', test_filename)
        print('N:', N)
        print('lambda:', lam)
        print('sep_order:', sep_order)
        print('asc_order',asc_order)
        print('maxsat:', maxsat)
        print('sep:', sep)
        print('ml:', ml)
        print('sparse:', sparse)

        benchmark = []
        benchmark.append(['Filename', 'Items', 'Features', 'Class', 'Solved', 'Nodes', 'Rule', 'Literal', 'Train_accuracy',
                          'Test_accuracy', 'Runtime'])


        Solve(train_filename, test_filename, N, lam, s_order=sep_order, asc_order = asc_order, maxsat=maxsat, sep=sep, ml=ml, sparse=sparse)


        #for rec in det:
         #   benchmark.append(rec)

        #print(benchmark)
        '''
        with open('sparse/aaa.csv', "w", newline="") as f:
            writer = csv.writer(f) #
            writer.writerows(benchmark)
            f.close()
        '''