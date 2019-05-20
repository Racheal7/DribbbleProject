# Metadata.csv

The csv file contains all metadata of **240,000** UI designs crawled from the design sharing community [Dribbble](https://dribbble.com/).

Each column corresponds to:
* ID: The unique id assigned by Dribbble when the design was uploaded.
* URL: Link to the corresponding web page.
* Designer’s name.
* Company or design team name: The design company or team the designer works for (may be None).
* Upload date: The date that the design was uploaded onto the website.
* Tags: A list of sematic tags specified by the designer when the design was uploaded.
* Number of views.
* Number of likes.
* Number of saves.
* Number of comments.
* Number of attachments.
* Title of the design (specified by the designer).
* A short description of the design.

# Comments.csv

This csv file contains all comments associated with the designs in the dataset. The structure of the Comments.csv is as follows:
* ID: The unique id assigned by Dribbble when the design was uploaded.
* Name: The name of the person who posted the comment
* Number of likes.
* Content: The content of each comment.
