import csv
from bs4 import BeautifulSoup
import re
import time
import threadpool
import queue
import os
import Requester

# Number of Thread
threadNum = 1
# Use a proxy or not
useProxyPool = False
# Whether to get the proxy from Api and use
useProxyFun = False
# ip pool
ipPool = []

# Build a basic request header
basicHeaders = {
    'accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'accept-encoding':'gzip, deflate, br',
    'accept-language':'zh-CN,zh;q=0.9',
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
 }
# Build a xml request header
xmlHeaders = basicHeaders['x-requested-with']="XMLHttpRequest"

requester = Requester.Requester(header = basicHeaders,useProxyPool=useProxyPool,useProxyFun=useProxyFun,ipPool=ipPool)
xmlRequester = Requester.Requester(header = xmlHeaders,useProxyPool=useProxyPool,useProxyFun=useProxyFun,ipPool=ipPool)

pool = threadpool.ThreadPool(threadNum)

# Seedlist - a list of IDs
r = range(4345000, 4345003) 
taskList = []
for id in r:
    url = 'https://dribbble.com/shots/%d'%id
    taskList.append(url)
    
def crawlId():
    requests = threadpool.makeRequests(crawlDetail, taskList)
    for req in requests:
        pool.putRequest(req)
    pool.wait()

## This function crawls all the required metadata.
def crawlDetail(link, mode=True):

    global taskList
    
    print("\n")

    pointSearch = re.compile('shots/(\d+)').findall(link)
    if len(pointSearch) == 0:
        point = int(time.time())
    else:
        point = pointSearch[0]


    if os.path.exists("All_images"+"/" + str(point) + ".gif"):
        return
    if os.path.exists("All_images"+"/" + str(point) + ".jpg"):
        return
    if os.path.exists("All_images"+"/" + str(point) + ".png"):
        return

    print(link)

    responseItem = requester.sendNewRequest(link)

    detailHtml = responseItem.response.content.decode("utf-8")

    if "Sorry, the page you were looking for doesn" in detailHtml:
        print("Sorry, the page you were looking for doesn't exist")
        return

    soup = BeautifulSoup(detailHtml,"html.parser")

    # Download image
    imgLinkTag = soup.find("div",attrs={'class':'detail-shot'})
    if imgLinkTag:
        imgLink = imgLinkTag['data-img-src']
        print("Downloading...", imgLink)
        if ".gif" in imgLink:
            with open("All_images/" + str(point) + ".gif", mode="wb") as f:
                imgSource = requester.sendNewRequest(imgLink).response.content
                f.write(imgSource)
                f.close()
        elif ".jpg" in imgLink:
            with open("All_images/" + str(point) + ".jpg", mode="wb") as f:
                imgSource = requester.sendNewRequest(imgLink).response.content
                f.write(imgSource)
                f.close()
        elif ".png" in imgLink:
            with open("All_images/" + str(point) + ".png", mode="wb") as f:
                imgSource = requester.sendNewRequest(imgLink).response.content
                f.write(imgSource)
                f.close()

    # Download title.
    titleSearch = soup.find("h1", attrs = {'class': "shot-title"})
    title = ""
    if titleSearch:
        title = str(titleSearch.get_text())
    
    # Download short description    
    desSearch = soup.find('div', attrs = {'class':"shot-desc"})
    description = ""
    if desSearch:
        description = str(desSearch.get_text()).replace("\n", " ")
    
    # Download designer
    byIdSearch =re.compile('by <a class="url hoverable shot-hoverable" rel="contact" href="/(.*?)">').findall(detailHtml)
    byId = ""
    if len(byIdSearch)!=0:
        byId = byIdSearch[0]

    # Download company or team name
    forIdSeach = re.compile('for <a class="hoverable shot-hoverable" rel="contact" href="/(.*?)">').findall(detailHtml)
    forId = ""
    if len(forIdSeach) != 0:
        forId = forIdSeach[0]

    # Download upload time.
    onTimeSearch = re.compile('<a href="/shots\?date=(.*?)">').findall(detailHtml)
    onTime = ""
    if len(onTimeSearch)!=0:
        onTime = onTimeSearch[0]

    # Download a list of tags.
    shotTagsTag = soup.find('div', attrs={'class': 'shot-tags'})
    shotTags = ""
    if shotTagsTag:
        shotTags = str(shotTagsTag.get_text()).replace("\n"," ")

    # Number of views
    viewsSearch = re.compile(' (.*?) views').findall(detailHtml)
    views = "0"
    if len(viewsSearch) != 0:
        views = viewsSearch[0]

    # Number of favorate
    favSearch = re.compile(' (\d+) likes').findall(detailHtml)
    fav = "0"
    if len(favSearch) != 0:
        fav = favSearch[0]

    # Number of saves
    savesSearch = re.compile(' (\d+) saves').findall(detailHtml)
    saves = "0"
    if len(savesSearch) != 0:
        saves = savesSearch[0]

    # Download attachments
    attachmentSearch = re.compile('src="https://cdn.dribbble.com/users/(.*?)/thumbnail/Attachment').findall(detailHtml)
    attachmentNumber = len(attachmentSearch)
    num = 1
    for attachmentItem in attachmentSearch:
        attachmentImgLink = "https://cdn.dribbble.com/users/{}/Attachment_{}".format(attachmentItem,str(num))
        print("Downloading attachments...", attachmentImgLink)
        with open("All_images/" + str(point)+"_"+str(num) + ".png", mode="wb") as f:
            imgSource = requester.sendNewRequest(attachmentImgLink).response.content
            f.write(imgSource)
            f.close()
        num += 1

    # Download comments
    commentLink = link + "/comments"
    responseItem = xmlRequester.executeOldRequest(commentLink, responseItem.session)
    response = responseItem.response
    commentHtml = response.content.decode('utf-8')
    cmnt = crawlComment(point, commentHtml)
    commentPage = int(int(cmnt) / 25) + 1
    for curPage in range(2, commentPage + 1):
        commentLink = link + "/comments?comments_sort=oldest&page=%d" % curPage
        responseItem = xmlRequester.executeOldRequest(commentLink, responseItem.session)
        response = responseItem.response
        commentHtml = response.content.decode('utf-8')
        crawlComment(point, commentHtml)

    dataList = [point, link,byId,forId,onTime,shotTags, views, fav, saves, cmnt,attachmentNumber, title, description]

    print(dataList)
    # Write metadata into a csv file
    csvFile = open("Metadata.csv", mode="a", encoding="utf-8", newline="")
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(dataList)
    csvFile.close()
    
    # Push new links into the frontier.
    if mode:
        moreTag = soup.find('ol',attrs={'class':'more-thumbs'})
        if moreTag:
            moreLinkSearch = re.compile('<a href="(.*?)">').findall(str(moreTag))
            if len(moreLinkSearch)!=0:
                moreLink = "https://dribbble.com"+moreLinkSearch[0]
                taskList.append(moreLink)
                
## This function crawls all the comments on the page.
def crawlComment(point,html):
    csvFile = open("Comments.csv", mode="a", encoding="utf-8",newline="")
    csvWriter = csv.writer(csvFile)
    cmnt = 0
    cmntSearch = re.compile(' (\d+) Responses').findall(html)
    if len(cmntSearch)!=0:
        cmnt = str(cmntSearch[0])

    soup = BeautifulSoup(html,"html.parser")

    allCommentTag = soup.find_all("li",attrs={'class':'response'})
    
    for commentTag in allCommentTag:
        Dsoup = BeautifulSoup(str(str(commentTag)),"html.parser")
        
        ## Poster of each comment
        nameTag = Dsoup.find("a")
        name = ""
        if nameTag:
            name = nameTag.get_text().replace("\n","")
        
        ## The number of likes of each comment.
        likeTag = Dsoup.find("a",attrs={"class":"likes-list"})
        likes = ""
        if likeTag:
            likes = likeTag.get_text().replace("\n","")  
        posted = ""
        postedTag = Dsoup.find("a",attrs={'class':'posted'})
        if postedTag:
            posted = postedTag.get_text().replace("\n","")
        
        ## Crawl the content of comments
        comment = ""
        cT = Dsoup.find("div",attrs={'comment-body'})
        if cT:
            comment = cT.get_text().replace("\n","")

        print(name,likes,comment,posted)
        csvWriter.writerow([point,name,likes,comment])
    csvFile.close()
    return cmnt


if __name__ == "__main__":
    crawlId()
