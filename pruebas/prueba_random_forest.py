# Install TensorFlow Decision Forests
#!pip install tensorflow_decision_forests

# Load TensorFlow Decision Forests
import tensorflow_decision_forests as tfdf

# Load the training dataset using pandas
import pandas
train_df = pandas.read_csv("penguins_train.csv")

# Convert the pandas dataframe into a TensorFlow dataset
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="species")

# Train the model
model = tfdf.keras.RandomForestModel()
model.fit(train_ds)