from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

shili = [
        'What is a general rule of thumb for sobriety and cycling? Couple hours after you "feel" sober? What? Or should I just work with "If I drink tonight, I do not ride until tomorrow"? ',
        '1 hour drink for the first 4 drinks.',
        '1.5 hours drink for the next 6 drinks.',
        '2 hours drink for the rest.'
]

tz = CountVectorizer()
tz1 = CountVectorizer(stop_words='english', decode_error='ignore')
X = tz.fit_transform(shili)
X1 = tz.fit_transform(shili)

print(X.shape,X1.shape)
print(X.toarray(),'\n\n',X1.toarray())
print('-'*100)
tz3 = TfidfVectorizer()
Y1 = tz3.fit_transform(shili)

tz4 = TfidfVectorizer(stop_words='english', decode_error='ignore')
Y2 = tz4.fit_transform(shili)
print(Y1.shape,Y2.shape)
print(Y1.toarray(),'\n\n',Y2.toarray())