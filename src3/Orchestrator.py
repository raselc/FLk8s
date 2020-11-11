'''
Created on november 08,2020
Modified on ,2020
@author: Rasel Chowdhury
@contributors: Sawsan AR
'''


def updateResource(deployName,namespace,cpu,memory):
    capi = client.AppsV1Api()
    deployment = capi.read_namespaced_deployment(deployName, namespace)
    deployment.spec.template.spec.containers[0].resources.limits['cpu'] = '0.5'
    deployment.spec.template.spec.containers[0].resources.limits['memory'] = '500Mi'
    print(deployment.spec.template.spec.containers[0].resources.limits)
    api_response= capi.patch_namespaced_deployment(name=deployName,namespace = namespace, body = deployment)
    print("Deployment updated. status='%s'" % str(api_response.status))
    #metadata = deployment.metadata.annotations.get('kubectl.kubernetes.io/last-applied-configuration')

def test():
    updateResource('client1','fedlearn','0.5','500')

if __name__ == "__main__":
    #main()
    test()