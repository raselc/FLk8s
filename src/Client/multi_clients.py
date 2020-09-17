import os
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num', help='number of client')
    args = parser.parse_args()
    print(args.num)
    i = 1
    j = int(args.num) + i

    for i in range(i, j):
        cmd = "start cmd /c python C:/Users/...***RASEL_PATH****..../FL_ResourceAllocation/" \
              "Experiments/nClients/RandomClientSelection/fl_client.py " + str(i)
        cmd1= "python "+os.getcwd()+"/fl_client.py " +str(i)
        cmd2= "python fl_client.py "+str(i)
        print(cmd1)
        print(os.getcwd())
        #os.system(cmd1)
        subprocess.Popen(cmd2,shell=True)



        '''subprocess.Popen(
            "python C:/Users/.../"
            "Experiments/nClients/fl_client.py 1")'''

