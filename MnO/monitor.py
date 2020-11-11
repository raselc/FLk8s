'''
Created on jan 10,2020
Modified on july 02,2020
@author: Rasel Chowdhury
'''

from kubernetes import client, config
from prometheus_http_client import Prometheus
import os
import json
import csv
import datetime
import time
import socketio

config.load_kube_config()

os.environ["PROMETHEUS_URL"] = 'http://localhost:9090'
debug = False
# debug= True
nameSpace = "test"
mList = []
sio = socketio.Client()


def getNodeInfo():
    nodeInfo = {}
    v1 = client.CoreV1Api()
    if debug: print(v1.list_node().items)
    for i in v1.list_node().items:
        if debug: print(i.metadata.name, i.status.allocatable['cpu'], i.status.allocatable['memory'])
        mem = i.status.allocatable['memory'][:-2]
        nodeInfo[i.metadata.name] = [i.status.allocatable['cpu'], mem, i.status.addresses[0].address]
    return (nodeInfo)


def getPodList():
    podInfo = {}
    v1 = client.CoreV1Api()
    b = v1.list_pod_for_all_namespaces(watch=False)
    #print(b.items)
    nodeinfo = getNodeInfo()
    for i in b.items:
        if (i.metadata.namespace == nameSpace and i.status.phase == 'Running'):
            if debug: print(i.metadata.name, i.metadata.namespace, i.spec.node_name, i.spec.containers[0].name)
            try:
                if ('app' in i.metadata.labels):
                    podInfo[i.metadata.name] = [i.metadata.namespace, i.spec.node_name, i.metadata.labels.get('app'),
                                                i.spec.containers[0].resources.limits.get('cpu'),
                                                int(i.spec.containers[0].resources.limits.get('memory')[:-2])]
                else:
                    podInfo[i.metadata.name] = [i.metadata.namespace, i.spec.node_name, i.spec.containers[0].name,
                                                i.spec.containers[0].resources.limits.get('cpu'),
                                                int(i.spec.containers[0].resources.limits.get('memory')[:-2])]
            except:
                if ('app' in i.metadata.labels):
                    podInfo[i.metadata.name] = [i.metadata.namespace, i.spec.node_name, i.metadata.labels.get('app'),
                                                nodeinfo.get(i.spec.node_name)[0], nodeinfo.get(i.spec.node_name)[1]]
                else:
                    podInfo[i.metadata.name] = [i.metadata.namespace, i.spec.node_name, i.spec.containers[0].name,
                                                nodeinfo.get(i.spec.node_name)[0], nodeinfo.get(i.spec.node_name)[1]]

    return podInfo


def cleanJson(val):
    c = val["data"]
    if not c["result"]:
        return ""
    else:
        d = c["result"][0]
        # print(d["value"][1])
        return d["value"][1]


def getCpu(containerName, nodeCpu):
    if debug: print("inside container cpu")
    cpu = 0
    prometheus = Prometheus()
    qry = 'sum(rate (container_cpu_usage_seconds_total{pod="' + containerName + '",image!="",container!="POD"}[1m]))'
    temp = json.loads(prometheus.query(metric=qry))
    if debug: print(temp)
    if temp == "":
        return 'N/A'
    else:
        value = cleanJson(temp)
        try:
            val = float(value)
            # cpu = val / int(nodeCpu) * 100
            cpu = val
        except:
            cpu = 'N/A'
        # print(type(val),"  : ",val)

    if debug: print(cpu)
    return cpu


def getMemory(containerName, nodeMemory):
    if debug: print("inside container memory")
    memory = 0
    prometheus = Prometheus()
    qry = 'container_memory_usage_bytes{pod="' + containerName + '",image!="",container!="POD"}'
    temp = json.loads(prometheus.query(metric=qry))
    if debug: print(temp)
    if temp == "":
        return 'N/A'
    else:
        try:
            # memory = (float(cleanJson(temp))/1024) / int(nodeMemory) * 100
            memory = (float(cleanJson(temp)) / 1024)
        except:
            memory = 'N/A'
    if debug: print(memory)
    return memory


def getNetwork(containerName):
    sent = ""
    prometheus = Prometheus()
    qry = 'rate(container_network_receive_bytes_total{pod="' + containerName + '",image!=""}[1m])'
    # qry = 'rate(container_network_receive_bytes_total{pod="prometheus-deployment-5946547f64-gz6qp"}[60s])'
    temp = json.loads(prometheus.query(metric=qry))
    if debug: print(temp)
    received = cleanJson(temp)
    qry2 = 'rate(container_network_transmit_bytes_total{pod="' + containerName + '",image!=""}[1m])'
    temp = json.loads(prometheus.query(metric=qry2))
    if debug: print(temp)
    sent = cleanJson(temp)
    return [received, sent]


def getNodeCpu(nodeIp, cpuCount):
    prometheus = Prometheus()
    qry = 'sum(rate(node_cpu{instance="' + nodeIp + ':9100"}[1m]))'
    if debug: print(json.loads(prometheus.query(metric=qry)))
    temp = cleanJson(json.loads(prometheus.query(metric=qry)))
    if debug: print(temp)
    if temp == "":
        return 'N/A'
    else:
        return (float(temp))


def getNodemMmory(nodeIp):
    prometheus = Prometheus()
    qry = 'node_memory_Active{instance="' + nodeIp + ':9100"}'
    if debug: print(json.loads(prometheus.query(metric=qry)))
    temp = cleanJson(json.loads(prometheus.query(metric=qry)))
    if debug: print(temp)
    if temp == "":
        return 'N/A'
    else:
        return (float(temp) / 1024)


def getNodeNetwork(nodeIp):
    prometheus = Prometheus()
    qry = 'sum(rate(node_network_receive_bytes{instance="' + nodeIp + ':9100"}[1m]))'
    temp = cleanJson(json.loads(prometheus.query(metric=qry)))
    if debug: print(temp)
    if temp == "":
        rcv = 'N/A'
    else:
        rcv = temp
    qry = 'sum(rate(node_network_transmit_bytes{instance="' + nodeIp + ':9100"}[1m]))'
    temp = cleanJson(json.loads(prometheus.query(metric=qry)))
    if debug: print(temp)
    if temp == "":
        sent = 'N/A'
    else:
        sent = temp
    return [rcv, sent]

def writeToCsv(data, mode):
    with open('data.csv', mode) as f:
        writer = csv.writer(f)
        writer.writerow(data)


'''WEB SOCKET '''


@sio.on('senData')
def predict():
    if debug: print("sending data")
    if debug: print(mList)
    sio.emit('predict', mList)
    # sio.emit('predict', {'data':['slave3', '4', '3838472', '0.01363413259648473', '1824516.0', '116334.74522350886', '171638.18647857622', 'default', 'dummy', 'dummy-pod-7c947bc7c9-dqf5g', '0', '0', '0.006392900075996606', '0.04262112632318277', '0', '0']})


@sio.on('disconnect')
def disconnect(a):
    if debug: print('disconnected from server ')
    # sio.emit('disconnect')
    sio.disconnect()


def main():
    global mList
    print(
        "NodeName  NumberOfCores  AllocatedMemory NodeIP nodeCPU nodeMemory ByteReceived ByteSent Namespace DeploymentName "
        "ContainerID AssignedCPU AssignedMemory  CPUusage  MemoryUsage  BytesReceived  BytesSent")
    i = 0
    end_t = time.time() + 60 * 60
    while (i<100):
        # while time.time() < end_t:
        mList = []
        nodeInfo = getNodeInfo()
        if debug: print(nodeInfo)
        podList = getPodList()
        # print(podList)
        for key in podList:
            podName = key
            nameSp = podList.get(key)[0]
            nodeName = podList.get(key)[1]
            cpuCount = nodeInfo.get(nodeName)[0]
            memTot = nodeInfo.get(nodeName)[1]
            podcpu = getCpu(podName, cpuCount)
            podAssignedCpu = podList.get(key)[3]
            podAssignedMemory = podList.get(key)[4]
            podMem = getMemory(podName, memTot)
            podNet = getNetwork(podName)
            deployName = podList.get(key)[2]
            nodeIp = nodeInfo.get(nodeName)[2]
            nodeLcpu = getNodeCpu(nodeIp, int(cpuCount))
            nodeMemoryUsage = getNodemMmory(nodeIp)
            nodeNet = getNodeNetwork(nodeIp)
            print(nodeName, cpuCount, memTot, nodeLcpu, nodeMemoryUsage, nodeNet[0], nodeNet[1], nameSp, deployName,
                  podName, podAssignedCpu, podAssignedMemory, podcpu, podMem, podNet[0], podNet[1])
            writeToCsv(
                [datetime.datetime.now(), nodeName, cpuCount, memTot, nodeLcpu, nodeMemoryUsage, nodeNet[0], nodeNet[1],
                 nameSp, deployName, podName, podAssignedCpu, podAssignedMemory, podcpu, podMem, podNet[0], podNet[1]],
                  'a')
            #mList.append(
            #    [nodeName, cpuCount, memTot, nodeLcpu, nodeMemoryUsage, nodeNet[0], nodeNet[1], nameSp, deployName,
            #     podName, podAssignedCpu, podAssignedMemory, podcpu, podMem, podNet[0], podNet[1]])
            #try:
            #    sio.connect('http://localhost:5001')
            #except:
            #    print("hehe")
        print(i)
        i = i + 1
        # print(mList)

        print("----------------------------------------------------------------------------------------------\n\n\n")
        time.sleep(2)


def test():
    print(getPodList())
    '''
    # podNet = getNetwork("prometheus-deployment-5946547f64-gz6qp")
    # print(podNet)
    # print(getNodeInfo())
    # print(getPodList())
    #global mList
    
    mList = [
        ["slave1", "4", "3838472", "3.953090909090979", "811660.0", "187532.41654909088", "230905.33456121208", "adc",
         "grpc-th-adcingress", "grpc-th-adcingress-envoy-769677ffb7-d8vk6", "4", "3838472", "0.013041770594369789",
         "15992.0", "16868.23672217569", "36524.68538741358", "None", "9.727272727272727", "None", "9.727272727272727",
         "2xx", "9.69090909090909", "2xx", "9.69090909090909", "284.25000000000006", "129.6"]]
    sio.connect('http://10.0.2.30:5001')
    '''


if __name__ == "__main__":
    writeToCsv(['time', 'Node Name', 'CPU Count', 'Allocated Memory', 'Node CPU', 'Node Memory', 'Node Network Receive',
                'Node Network sent', 'NameSpace', 'Deployment Name', 'Pod Name', 'Assigned Container CPU',
                'Assigned Container Memory', 'Pod CPU', 'Pod Memory', 'Bytes Received', 'Bytes Sent'], 'w')
    main()
