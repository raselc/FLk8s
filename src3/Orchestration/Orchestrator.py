'''
Created on november 08,2020
Modified on ,2020
@author: Rasel Chowdhury
@contributors: Sawsan AR
'''

from flask import *
import socketio
from flask_socketio import SocketIO,emit, send
from kubernetes import client, config

'''Global Vars'''
config.load_kube_config()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'LifeIsBeautiful'
socio = SocketIO(app)
debug = True
namespace = 'fedlearn'

'''CLIENT HANDLER'''
@socio.on('connect')
def on_connect():
    emit('sendData', room=request.sid)

@socio.on('getResource')
def getResource(deployName):
    print(deployName)
    capi = client.AppsV1Api()
    deployment = capi.read_namespaced_deployment(deployName, namespace)
    cpu = deployment.spec.template.spec.containers[0].resources.limits['cpu']
    memory = deployment.spec.template.spec.containers[0].resources.limits['memory']
    print(cpu, memory)
    emit('resource',{'cpu':cpu,'memory':memory}, room=request.sid)


@socio.on('updateResource')
def updateResource(data):
    print(data)
    capi = client.AppsV1Api()
    deployName = data.get('name')
    cpu = data.get('cpu')
    memory = data.get('memory')
    deployment = capi.read_namespaced_deployment(deployName, namespace)
    deployment.spec.template.spec.containers[0].resources.limits['cpu'] = cpu
    deployment.spec.template.spec.containers[0].resources.limits['memory'] = memory+'Mi'
    print(deployment.spec.template.spec.containers[0].resources.limits)
    api_response= capi.patch_namespaced_deployment(name=deployName,namespace = namespace, body = deployment)
    print("Deployment updated. status='%s'" % str(api_response.status))
    #metadata = deployment.metadata.annotations.get('kubectl.kubernetes.io/last-applied-configuration')
    emit('dis', room=request.sid)


@socio.on('disconnect')
def disconnect():
    print("disconnected",request.sid)
    #socio.disconnect()

def main():
    if debug: print("inside main")
    socio.run(app,'0.0.0.0',51707)

def test():
    updateResource('client1','fedlearn','1','500')
    getResource('client1','fedlearn')

if __name__ == "__main__":
    main()
    #test()