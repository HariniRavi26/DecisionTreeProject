import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


# TITLE

st.title("💻 Laptop Purchase Prediction App")

st.write(
"Predict whether customer will buy a laptop"
)


# LOAD DATASET

df = pd.read_csv(
"dataset.csv"
)


# ENCODE DATA

encoders = {}

for col in df.columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col]
    )

    encoders[col] = le


# FEATURES + TARGET

X = df.drop(
"Buys_Laptop",
axis=1
)

y = df[
"Buys_Laptop"
]


# TRAIN MODEL

model = DecisionTreeClassifier()

model.fit(
X,
y
)


# USER INPUT

st.header(
"Enter Customer Details"
)


age = st.selectbox(

"Age",

[
"Youth",

"Middle",

"Senior"

]

)


income = st.selectbox(

"Income",

[
"Low",

"Medium",

"High"

]

)


student = st.selectbox(

"Student",

[
"Yes",

"No"

]

)


credit = st.selectbox(

"Credit Rating",

[
"Fair",

"Excellent"

]

)


# PREDICTION

if st.button(

"Predict"

):


    input_data = pd.DataFrame({

        "Age":[age],

        "Income":[income],

        "Student":[student],

        "Credit_Rating":[credit]

    })


    for col in input_data.columns:

        input_data[col] = encoders[
            col
        ].transform(

            input_data[col]

        )


    prediction = model.predict(

        input_data

    )[0]


    result = encoders[
        "Buys_Laptop"
    ].inverse_transform(

        [prediction]

    )[0]


    if result=="Yes":

        st.success(

        "✅ Customer Will Buy Laptop"

        )


    else:

        st.error(

        "❌ Customer Will NOT Buy Laptop"

        )

