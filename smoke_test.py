#!/usr/bin/env python3
import sys
import os

DATA_DIR = './' #TODO: change this to wherever you put the data if working on a different machine
SIGFIG_NUM = 5

def err(msg):
    print('ERROR: {}'.format(msg), file=sys.stderr) #NOTE: If you get a SyntaxError on this line, you are using Python 2, which is wrong. Use Python 3.
    exit()

def find_filenames():
    return [fn for fn in os.listdir('.') if os.path.isfile(fn) and (fn.endswith('_hmm_gaussian.py') or fn.endswith('_hmm_aspect.py'))]

def get_output(filename):
    import subprocess
    cmd = None
    if filename.endswith('_hmm_gaussian.py'):
        cmd = 'python {} --nodev --iterations 2 --clusters_file gaussian_hmm_smoketest_clusters.txt --data_file {} --print_params'.format(filename, os.path.join(DATA_DIR,'points.dat'))
    else:
        cmd = 'python {} --nodev --iterations 2 --clusters_file aspect_hmm_smoketest_clusters.txt --data_file {} --print_params'.format(filename, os.path.join(DATA_DIR,'pairs.dat'))
    print('Running this command:\n{}'.format(cmd))
    try:
        output = subprocess.check_output(cmd.split()).decode('utf-8')
    except subprocess.CalledProcessError:
        err('Python file did not exit successfully (likely crashed).')
    except OSError as e:
        if e.errno == 13:
            err('Python file is not executable; run this command:\nchmod u+x {}'.format(filename))
        elif e.errno == 8:
            err('Python file does not start with a shebang; put this line at the very top:\n#!/usr/bin/python3')
        elif e.errno == 2:
            err('Unable to execute python file; if you ever edited the file on Windows, it is possible that the line endings are wrong, so try running this command:\ndos2unix {}\nOtherwise, you\'ll have to take a look at it and see what went wrong.'.format(filename))
        else:
            print('Got an OS error not caused by permissions or a shebang problem; you\'ll have to take a look at it and see what the problem is. See below:')
            raise e

    return output

def tokens(s):
    result = []
    for tok in s.split():
        try:
            result.append(float(tok))
        except ValueError as e:
            result.append(tok)
    return result

def round_to_sigfigs(num, sigfigs):
    from math import log10, floor
    if num == 0:
        return num
    else:
        return round(num, -int(floor(log10(abs(num)))) + sigfigs - 1)

def fuzzy_match(line, req):
    line_toks = tokens(line)
    if len(line_toks) != len(req):
        return False
    else:
        for l,r in zip(line_toks, req):
            if type(l) != type(r):
                return False
            elif type(l) == str and l != r:
                return False
            elif type(l) == float and round_to_sigfigs(l,SIGFIG_NUM) != round_to_sigfigs(r,SIGFIG_NUM): #float
                return False
        return True

class Req:
    def __init__(self, req, name):
        self.req = tokens(req)
        self.name = name
        self.matched = False

    def check(self,line):
        if fuzzy_match(line, self.req):
            self.matched = True

    def report(self):
        s = '{}: '.format(self.name)
        if self.matched:
            return s + 'Correct!'
        else:
            return s + 'NOT CORRECT!'

    def req_str(self):
        return ' '.join(map(str,self.req))


def verify_reqs(reqs, output):
    for line in output.split('\n'):
        for r in reqs:
            r.check(line)
    for r in reqs:
        print(r.report())
    if not all([r.matched for r in reqs]):
        err('Unable to find one or more required output lines. Make sure each is on its own line and formatted correctly; if so, then there is an implementation problem. This should have produced (with all numbers matched to {} significant figures):\n{}\n'.format(SIGFIG_NUM, '\n'.join([r.req_str() for r in reqs])))

def main():
    filenames = find_filenames()
    if len(filenames) == 0:
        err('No files ending in \'_hmm_gaussian.py\' or \'_hmm_aspect.py\' found. Make sure your file is named LastName_hmm_gaussian.py or LastName_hmm_aspect.py.')
    if len(filenames) > 1:
        err('Only include a single file ending in \'_hmm_gaussian.py\' or \'_hmm_aspect.py\' in the submission directory.')
    print('Found Python file to run.')
    if not os.path.exists(DATA_DIR):
        err('Could not find the data directory; looked for {}. Change the DATA_DIR variable at the top of this smoke test file to wherever you have downloaded the data (points.dat or pairs.dat).'.format(DATA_DIR))
    print('Found data directory.')
    output = get_output(filenames[0])
    print('Ran Python file.')

    reqs = None
    if filenames[0].endswith('_hmm_gaussian.py'):
        reqs = [
            Req('Gaussian','Choice of Gaussian vs. Aspect'),
            Req('Train LL: -4.7420480591544445', 'Training average log-likelihood'),
            Req('Initials: 4.545815416616572e-06 | 0.9999954541845834','Initials'),
            Req('Transitions: 0.6517819332450636 0.34821806675493633 | 0.2893844133102026 0.7106155866897974', 'Transitions'),
            Req('Mus: -0.7516395901823867 -0.48006048420065606 | -0.8930856583800232 -0.6875453864106934', 'Mus'),
            Req('Sigmas: 2.0623041297020404 0.8044420623712198 0.8044420623712198 7.525809563367458 | 12.482218753578287 -0.9602732355782989 -0.9602732355782989 5.299079300138309', 'Sigmas')
            ]
    else:
        reqs = [
            Req('Aspect','Choice of Gaussian vs. Aspect'),
            Req('Train LL: -4.49323096289567', 'Training average log-likelihood'),
            Req('Initials: 0.023166708216632487 | 0.9768332917833674','Initials'),
            Req('Transitions: 0.5957849894090658 0.4042150105909342 | 0.4073607783018307 0.5926392216981693','Transitions'),
            Req('Theta_1: 0.08987511040144469 0.19471141042562856 0.037878648516166895 0.17756798758886985 0.10250331994662923 0.06834348298821478 0.0865174097539417 0.07171344141113004 0.1147857019091789 0.05610348705879503 | 0.16586924024149008 0.12734344702033468 0.07777602560037052 0.09110610076856249 0.15095037222873725 0.027109633758122642 0.10018320736995698 0.05268372679445988 0.17202316194154557 0.034955084276419725','Theta_1'),
            Req('Theta_2: 0.15040858502004908 0.18163851463660313 0.05873757755055805 0.060041481264120386 0.06026242954291614 0.10008451571801735 0.11098449485049505 0.08677876855109937 0.09227780265931429 0.09878583020682662 | 0.07830116468069032 0.09815408183342587 0.10582384508697326 0.10228568984121109 0.11320242461995303 0.16006449480973603 0.0911885475520775 0.08209849295232577 0.08771085149138619 0.08117040713222057','Theta_2')
                ]
    verify_reqs(reqs,output)
    print('Congratulations, you passed this simple test! However, make sure that your code runs AND PASSES THIS TEST on the csug server. Also, don\'t forget your README!')

if __name__ == '__main__':
    main()
