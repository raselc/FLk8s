'''
Created on feb 12,2020
Modified on ,2020
@author: Rasel Chowdhury
@contributors:
'''
import sys
import time
import os
from flask import *
import socketio
from flask_socketio import SocketIO,emit, send
from kubernetes import client, config


debug = True
#debug = False


'''GLOBAL'''
app = Flask(__name__)
app.config['SECRET_KEY'] = 'LifeIsBeautiful'
socio = SocketIO(app)
app.debug = True
config.load_kube_config()

'''CLIENT HANDLER'''
@socio.on('connect')
def on_connect():
    if debug: print("Connected to ", request.sid)
    emit('letStart',room=request.sid)

@socio.on('scaleup')
def scaleUp(cmd):
    if debug: print("doing scaleUp")
    if debug: print(cmd)
    capi = client.AppsV1Api()
    #print(capi.read_namespaced_deployment(cmd.get('deployName'), cmd.get('nameSpace')))
    try:
        currentReplica = capi.read_namespaced_deployment(cmd.get('deployName'), cmd.get('nameSpace')).spec.replicas
    except:
        capi = client.CustomObjectsApi()
        currentReplica = capi.get_namespaced_custom_object('adc.ericsson.com','v1alpha1',cmd.get('nameSpace'),'adcingresses',cmd.get('deployName')).get('spec').get('replicas')
    #print(currentReplica)
    #currentReplica = 2
    print("scaling up")
    customObject(cmd.get('deployName'), cmd.get('nameSpace'), int(currentReplica)+int(cmd.get('replica')))
    #if debug: print("all done for ",request.sid)
    socio.emit('disconnect',"")
    

@socio.on('scaledown')
def scaledown(cmd):
    if debug: print(cmd)
    capi = client.AppsV1Api()
    #print(capi.read_namespaced_deployment(cmd.get('deployName'), cmd.get('nameSpace')))
    try:
        currentReplica = capi.read_namespaced_deployment(cmd.get('deployName'), cmd.get('nameSpace')).spec.replicas
    except:
        capi = client.CustomObjectsApi()
        currentReplica = capi.get_namespaced_custom_object('adc.ericsson.com','v1alpha1',cmd.get('nameSpace'),'adcingresses',cmd.get('deployName')).get('spec').get('replicas')
    #print(currentReplica)
    #currentReplica = 2
    print(currentReplica,"\n\n\n")
    if currentReplica > 1 :
        print("scaling down")
        customObject(cmd.get('deployName'), cmd.get('nameSpace'), int(currentReplica)-int(cmd.get('replica')))
    #if debug: print("all done for ",request.sid)
    socio.emit('disconnect',"")

@socio.on('disconnect')
def disconnect():
    print("disconnected",request.sid)
    #socio.disconnect()
    
    

def customObject(deployName,namespace, replicaNumber):
    config.load_kube_config
    v1 = client.CustomObjectsApi()
    v2 = v1.get_namespaced_custom_object('adc.ericsson.com','v1alpha1',namespace,'adcingresses',deployName)
    v2.get('spec')['replicas'] = replicaNumber
    api_response = v1.patch_namespaced_custom_object(group='adc.ericsson.com',version='v1alpha1',namespace=namespace,plural='adcingresses',name=deployName,body=v2)
   
    print(api_response)
    

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def scaleDep(deployName,namespace, replicaNumber):
    
    v1 = client.AppsV1Api()
    v2 = v1.read_namespaced_deployment(deployName, namespace)
    v2.spec.replicas = replicaNumber
    api_response = v1.patch_namespaced_deployment( name=deployName, namespace=namespace, body=v2)
    if debug: print("Deployment updated. status='%s'" % str(api_response.status))
    if debug: print("scaling DONE :P")

def updateResource(deployName,namespace,cpu,memory):
    capi = client.AppsV1Api()
    deployment = capi.read_namespaced_deployment(deployName, namespace)
    deployment.spec.template.spec.containers[0].resources.limits['cpu'] = '0.5'
    deployment.spec.template.spec.containers[0].resources.limits['memory'] = '500Mi'
    print(deployment.spec.template.spec.containers[0].resources.limits)
    api_response= capi.patch_namespaced_deployment(name=deployName,namespace = namespace, body = deployment)
    print("Deployment updated. status='%s'" % str(api_response.status))
    #metadata = deployment.metadata.annotations.get('kubectl.kubernetes.io/last-applied-configuration')



def main():
    if debug: print("inside main")
    socio.run(app,'localhost',5000)
    #test()
    

def test():
    #scaleUp({'deployName':'grpc-th-adcingress','nameSpace':'adc','replica':'1'})
    #scaledown({'deployName':'grpc-th-adcingress','nameSpace':'adc','replica':'1'})
    updateResource('client1','fedlearn','0.5','500')


if __name__ == "__main__":
    #main()
    test()