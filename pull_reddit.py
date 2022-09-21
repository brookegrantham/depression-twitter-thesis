import pandas as pd
from pmaw import PushshiftAPI
import datetime as dt

api = PushshiftAPI()


before = int(dt.datetime(2022,2,1,0,0).timestamp())
after = int(dt.datetime(2010,12,1,0,0).timestamp())

limit = 50000

posts=[]
subreddit_arr = ['depression','suicidewatch','anxiety','bpd','lonely','ptsd','opiates','offmychest','bipolarreddit','sad','TrueOffMyChest']
for sub in subreddit_arr:
  print(len(posts))
  posts += api.search_submissions(subreddit=sub, limit=limit,before=before,after=after)
posts_df = pd.DataFrame(posts, columns=['selftext'])

posts_df.to_csv('reddit-pretrain-text', index=None,header=False)
