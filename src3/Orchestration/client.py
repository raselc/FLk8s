import socketio

def verify(client):
    sOrches = socketio.Client()
    ipOrches = 'localhost'
    res = {'cpu':'0','memory':'0'}
    print(type(res))
    print("1")
    #res['cpu'] = '00'

    @sOrches.on('sendData')
    def data():
        print("2")
        sOrches.emit('getResource',client)

    @sOrches.on('resource')
    def resource(data):
        print("3")
        print (data)
        res['cpu'] = data.get('cpu')
        res['memory'] = data.get('memory')
        #requiredResource = predict()
        #cpu = requiredResource.get('cpu')
        #memory = requiredResource.get('memory')
        #sOrches.emit('updateResource', {'name': client, 'cpu': '1.2', 'memory': "400"})
        sOrches.disconnect()



    def predict():
        print("4")
        res = {'cpu':'0.3','memory':'400'}
        return res

    sOrches.connect('http://'+ipOrches+':51707')
    sOrches.wait()
    print("5")
    print(type(res))
    return res


def getResource(client):
    sOrches = socketio.Client()
    ipOrches = 'localhost'
    res = {'cpu':'','memory':''}
    print(type(res))
    print("1")


    @sOrches.on('sendData')
    def data():
        print("2")
        sOrches.emit('getResource', client)

    @sOrches.on('resource')
    def resource(data):
        print("3")
        print(data)
        sanitize(data)
        #res = data
        sOrches.disconnect()

    def sanitize(data):

        if data['cpu'] == '1' or data['cpu'] == '2':
            cpu = int(data['cpu'])
        else:
            cpu = int(data['cpu'][:-1]) / 1000
        memory = int(data['memory'][:-2])
        print("cpu:", cpu)
        print("memory:", memory)
        res['cpu'] = cpu
        res['memory'] = memory

    sOrches.connect('http://' + ipOrches + ':51707')
    sOrches.wait()
    print("5")
    print(type(res))
    return res



def updateResource(client,res):
    sOrches = socketio.Client()
    ipOrches = 'localhost'
    print(res)
    @sOrches.on('sendData')
    def data():
        print("2")
        sOrches.emit('updateResource', {'name': client, 'cpu': res['cpu'], 'memory': res['memory']})
    @sOrches.on('dis')
    def dis():
        sOrches.disconnect()
    sOrches.connect('http://' + ipOrches + ':51707')
    sOrches.wait()


if __name__ == "__main__":
    #a = getResource('client1')
    #print("6")
    #print(a)
    updateResource('client1',{'cpu':'0.3','memory':'400'})
    print(getResource('client1'))

    #main()

