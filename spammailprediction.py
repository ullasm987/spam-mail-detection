import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
load_model=pickle.load(open("C:/Users/sloka/OneDrive/Preparation/spam mail detection/train_model.sav"))


feature_ext=TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
input_text=["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive"]
fea_input_text=feature_ext.transform(input_text)
ans=load_model.predict(fea_input_text)
print(ans)

if ans[0]==0:
  print("spam mail")
else:
  print("Not a spam mail")