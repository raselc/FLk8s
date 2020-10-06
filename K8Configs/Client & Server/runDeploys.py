import os
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num', help='number of client')
    args = parser.parse_args()
    print(args.num)

    if args.num == 'start':
        os.system("kubectl apply -f test_c1.yaml")
        os.system("kubectl apply -f test_c2.yaml")
        os.system("kubectl apply -f test_c3.yaml")
        os.system("kubectl apply -f test_c4.yaml")
        os.system("kubectl apply -f test_c5.yaml")
        os.system("kubectl apply -f test_c6.yaml")
        print("All started")
    if args.num == 'end':
        os.system("kubectl delete -f test_c1.yaml")
        os.system("kubectl delete -f test_c2.yaml")
        os.system("kubectl delete -f test_c3.yaml")
        os.system("kubectl delete -f test_c4.yaml")
        os.system("kubectl delete -f test_c5.yaml")
        os.system("kubectl delete -f test_c6.yaml")
        print("deleted")