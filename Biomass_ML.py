from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
#%%

# Load the testing and data that shall be predicted
data = pd.read_csv('results.csv')
prediction_data=pd.read_csv('test.csv')

#%%

# Separate the features (X) and the targets (y)
X = data[['glc_bound', 'nh4_bound']]
y = data.drop(['glc_bound', 'nh4_bound', "murein4p4p_p", "chor_c", "pg161_c", "pe161_p", 
"murein4px4p_p", "pg161_p", "ni2_c", "malcoa_c", "udcpdp_c", "nadp_c", "5mthf_c", "bmocogdp_c", 
"adocbl_c", "2dmmql8_c", "succoa_c", "nh4_c", "q8h2_c", "enter_c", "pe160_p", "pg160_p", 
"pg160_c", "glycogen_c", "spmd_c", "ptrc_c", "gthrd_c", "clpn181_p", "ribflv_c", 
"mobd_c", "pg181_c", "pg181_p", "mocogdp_c", "murein3px4p_p", "murein3p3p_p", "clpn161_p", 
"mlthf_c", "colipa_e", "murein4px4px4p_p", "pe181_c", "pe181_p", "so4_c", "cl_c", "lipopb_c",
"clpn160_p", "mococdp_c", "mql8_c", ], axis=1)

 #%%
 
#Do the same for the prediction data
X_p = prediction_data[['glc_bound', 'nh4_bound']]
y_p = prediction_data.drop(['glc_bound', 'nh4_bound', "murein4p4p_p", "chor_c", "pg161_c", "pe161_p", 
"murein4px4p_p", "pg161_p", "ni2_c", "malcoa_c", "udcpdp_c", "nadp_c", "5mthf_c", "bmocogdp_c", 
"adocbl_c", "2dmmql8_c", "succoa_c", "nh4_c", "q8h2_c", "enter_c", "pe160_p", "pg160_p", 
"pg160_c", "glycogen_c", "spmd_c", "ptrc_c", "gthrd_c", "clpn181_p", "ribflv_c", 
"mobd_c", "pg181_c", "pg181_p", "mocogdp_c", "murein3px4p_p", "murein3p3p_p", "clpn161_p", 
"mlthf_c", "colipa_e", "murein4px4px4p_p", "pe181_c", "pe181_p", "so4_c", "cl_c", "lipopb_c",
"clpn160_p", "mococdp_c", "mql8_c", ], axis=1)

#%%

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

#%%

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)
y_p_scaled = scaler.transform(y_p)

#%%

#Introduce a learning rate decay
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, min_lr=0.00001, verbose=1)

#%%
# Define the model
model = Sequential()
model.add(Dense(y_train.shape[1], activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(y_train.shape[1]))

#%%

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#%%

# Train the model and save the history
history = model.fit(X_train, y_train_scaled, validation_data=(X_test, y_test_scaled),
                    epochs=10000, batch_size=32, callbacks=[reduce_lr])


#%%

# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#%%

# Predict the targets for the prediction data
y_p_scaled_pred = model.predict(X_p)

# Inverse transform the predicted targets
y_p_pred = scaler.inverse_transform(y_p_scaled_pred)


#%%

# Compute the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_p, y_p_pred)

# Compute the Mean Squared Error (MSE)
mse = mean_squared_error(y_p, y_p_pred)

# Compute the R-squared (R2) score
r2 = r2_score(y_p, y_p_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

#%%

#Create the scatteplot
plt.figure(figsize=(10, 6))
plt.scatter(y_p, y_p_pred)
plt.plot([y_p.min(), y_p.max()], [y_p.min(), y_p.max()], 'k--', lw=4)
plt.xlabel('ME-simulated values')
plt.ylabel('ML-predicted values')
plt.title('Scatter plot of ML vs. ME-model predicted values')
plt.show()


#%%

#Save the model
model.save('Biomass_prediction.h5')

#%%

# Load the saved model
model = load_model('Biomass_prediction.h5')


#%%

#Create data for the graphs
values = np.arange(-10, 0, 0.2)
nh4_bound = pd.DataFrame({
    'column1': -10,  
    'column2': values  
})
#%%

#Predict the data, inverse transform it and convert to panda dataframe
g_pred_scaled = model.predict(nh4_bound)
g_pred = scaler.inverse_transform(g_pred_scaled)
g_pred_df = pd.DataFrame(g_pred, columns=y_p.columns)

#%%

result = pd.concat([nh4_bound, g_pred_df], axis=1)

# Save the DataFrame to a CSV file
result.to_csv('g_values.csv', index=False)
