import yfinance as yf 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = yf.download("LT.NS", start="2020-01-01", end="2026-01-01")

df["Return"] = df['Close'].pct_change()
df["Yes_return"] = df["Return"].shift(1)

df = df.dropna()

df["Direction"] = (df["Return"] >0).astype(int)

x = df[["Yes_return"]]
y = df["Direction"]

x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.2)

model = LogisticRegression()
model.fit(x_train,y_train)

accuracy = model.score(x_test,y_test)
print("Models accuracy in %", accuracy*100)

plt.plot(x,y)

plt.show()




