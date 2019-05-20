import requests
import time
import random
import urllib3

urllib3.disable_warnings()
class Requester():

    def __init__(self,header,useProxyPool,useProxyFun,ipPool):
        self.header = header
        self.ipPool = ipPool
        self.useProxyPool = useProxyPool
        self.useProxyFun = useProxyFun

    def sendNewRequest(self,url,limit=3):
        reqNum = 0
        while True or reqNum<limit:
            try:
                req = requests.session()
                req.headers = self.header
                if self.useProxyPool:
                    req.proxies = {'http': random.choice(self.ipPool)}
                if self.useProxyFun:
                    proxyip = self.getIP()
                    req.proxies = {"http" : 'http://' + proxyip, "https" : 'https://' + proxyip}
                else:
                    req.proxies = None
                response = req.get(url, verify=False,timeout=5)
                requesterItem = RequesterItem()
                requesterItem.response = response
                requesterItem.session = req
                return requesterItem
            except requests.exceptions.ConnectionError:
                print('ConnectionError -- please wait 3 seconds')
                time.sleep(3)
            except requests.exceptions.ChunkedEncodingError:
                print('ChunkedEncodingError -- please wait 3 seconds')
                time.sleep(3)
            except:
                print('Unfortunitely -- An Unknow Error Happened, Please wait 3 seconds')
                time.sleep(3)
            finally:
                reqNum +=1
        return None

    def executeOldRequest(self,url,session,limit=3):
        reqNum = 0
        while True or reqNum < limit:
            try:
                if self.useProxyPool:
                    session.proxies = {'http': random.choice(self.ipPool)}
                elif self.useProxyFun:
                    proxyip = self.getIP()
                    session.proxies =  {"http" : 'http://' + proxyip, "https" : 'https://' + proxyip}
                else:
                    session.proxies = None
                response = session.get(url,verify=False,timeout=15)
                requesterItem = RequesterItem()
                requesterItem.response = response
                requesterItem.session = session
                return requesterItem
            except requests.exceptions.ConnectionError:
                print('ConnectionError -- please wait 3 seconds')
                time.sleep(3)
            except requests.exceptions.ChunkedEncodingError:
                print('ChunkedEncodingError -- please wait 3 seconds')
                time.sleep(3)
            except:
                print('Unfortunitely -- An Unknow Error Happened, Please wait 3 seconds')
                time.sleep(3)
            finally:
                reqNum += 1
        return None
    def getIP(self):
        apiUrl = "http://api.ip.data5u.com/dynamic/get.html?order=d213bc840cb7b4ebc186192091359e3e&sep=3"
        ipTxt = requests.get(apiUrl).content.decode()
        return ipTxt.split("\n")[0]


class RequesterItem():

    def __init__(self):
        self.response = None
        self.session = None