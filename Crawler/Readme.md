## Crawler Usage

These two files is used to crawl designs and metadata from [Dribbble](https://dribbble.com/). Follow these three steps to crawl data.
### Step 1: Download Requester.py and Spider.py

### Step 2: Set paths in Spider.py

Change the All_images to the name of your image folder.
```python
    # Download image
    imgLinkTag = soup.find("div",attrs={'class':'detail-shot'})
     ...
        with open("All_images/" + str(point) + ".gif", mode="wb") as f:
                ...
        with open("All_images/" + str(point) + ".jpg", mode="wb") as f:
                ...
        with open("All_images/" + str(point) + ".png", mode="wb") as f:
                ...
 ```
 Give a list of IDs as seed list. Note that the ids are six digits.
 ```python
     r = range(4345000, 4347500) 
 ```
 
 ### Step 3: Run the crawler
 You will find two csv files: Metadata.csv and Comments.csv.
