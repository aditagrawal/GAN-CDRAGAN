# Generative Adversarial Networks framework
This is Generative Adversarial Network framework written on Python using Tensorflow GPU package. If you decide to generate synthetic data from your current dataset, you should implement the following steps:

Add all paths and parameters in settings.json file
Preprocess data
Train GAN on preprocessed data
Generate new data
Postprocess data
Data format
Your data should be in CSV format with header or without header. Additionally, you should create the header file that contains name of the columns in the first line, type of the columns in the second line and default missing values in the third line. The example of data and header file can be found on GoldML cluster:

# Parameters in settings.json file
For your GAN project you should create settings.json file with the following parameters:

gan_pars: path to JSON file with GAN architecture description (see below sections)
viz_folder: path to the folder where histograms for Box-Cox transformation will be saved
error_file: path to the log file where you will be able to see the training generator and discriminator errors
model_folder: path to the folder where GAN model will be saved
laplace_pars: path to JSON file with column statistics necessary for Laplace transformation on categorical columns (optional)
onehot_pars: path to JSON file with one hot encoding statistics for categorical columns (optional)
boxcox_pars: path to JSON file with Box-Cox statistics for numerical columns (optional)
missing_pars: path to JSON file with statistics by missing values (optional)
special_pars: path to JSON file with statistics by special and missing values (optional)
scaling_pars: path to JSON file with normalization statistics (optional)
train_file: path to CSV training data
header_file: path to header for CSV training data with the column descriptions (K - for key column, C - for categorical column, N - for numerical column, L - for target column)
missing_train_file: path to CSV file that contains indicator variables for missing values
scaled_train_file: path to preprocessed CSV training data
scaled_header_file: path to the header for preprocessed CSV training data (notice: this file will be different from original header file)
gen_scaled_train_file: path to generated CSV file before postprocessing
gen_train_file: path to generated CSV file after preprocessing
header: 1 - if original training file contains header row, 0 - otherwise
path_to_library: path to the folder where utils.py and gan_net.py files are located
NOTE! JSON files will be created automatically when you run preprocess_data.py file (please check the next section in order to understand of how to preprocess your data).

# Preprocess data
You should select the preprocessing steps for your data before you train GAN. We have created several building blocks for preprocessing (all available building blocks for preprocessing can be found in utils.py file. For each use case you should use different buildign blocks for preprocessing.

For fraud use case we have selected the following preprocessing building blocks:

One hot encoding for categorical features
onehot = PandasOneHotEncoder(json_file=json_file_onehot)
train,colOnehot = onehot.fit_transform(train, colCategorical)
This building block will check which categorical features require one-hot encoding and convert these features to the binary features. It will generate JSON file specified in onehot_pars parameter in the settings.json file.

Missing and special values treatment (in case if your data have missing values)
mtr = PandasSpecialTreatment(json_file=json_file_special)
train,colMissing,colHighfreq = mtr.fit_transform(train, colNumerical)
This can be replaced by

mtr = PandasMissingTreatment(json_file=json_file_missing)
train,colMissing = mtr.fit_transform(train, colNumerical)
The main difference between PandasSpecialTreatment and PandasMissingTreatment is in treating the high frequency values. For several features in fraud data, there are a lot of observations with the same value (for example, zero value). If you apply special treatment, this block will create separate binary variable for such values. JSON files specified in missing_pars or special_pars will be created in the output folder.

Box-Cox transformation for numerical features
bcx = PandasBoxCox(json_file=json_file_boxcox)
train = bcx.fit_transform(train, colNumerical, visualize=1)
This building block will fit Box-Cox transformaton for all numerical features and save the results to the JSON file specified in boxcox_pars parameter in the settings.json file. We advise to set flag visualize to 1, it will generate comparison in distributions in the visualization folder specified in viz_folder parameter in the settings.json file.

Normalization of all features
scl = PandasStandardScaler(json_file=json_file_scale)
train = scl.fit_transform(train, colNumerical+colOnehot)
This building block will scale all features and creates JSON file specified in scaling_pars parameter in the settings.json file.

Replacing missing values
train = replace_missing_values(train, colNumerical, method='median')
This method will replace all missing values by median.

### IMPORTANT NOTICE! After you run preprocess_data.py with fit_boxcox parameter set to 1, you should manually review Box-Cox transformed features and correct lambda parameter (setting this parameter to zero means log transformation). After you correct JSON Box-Cox file, you can set fit_boxcox in the settings.json file to zero and run preprocess_data.py again. The method transform will be applied for Box-Cox transformation:

bcx = PandasBoxCox(json_file=json_file_boxcox)
train = bcx.transform(train, colNumerical)

# Train GAN on preprocessed data
Currently, we have the following GAN algorithms implements:

DRAGAN
Conditional DRAGAN
CycleGAN
CycleDRAGAN
Wasserstein GAN

The Python classes for these algorithms can be found in the gan_net.py file. The GAN arcitecture should be described in JSON file:

{
    "generator_layers": {
        "input": {
            "type": "dense",
            "size": 50,
            "previous": ""
        },
        "layer1": {
            "type": "dense",
            "previous": "input",
            "size": 128
        },
        "layer2": {
            "type": "relu",
            "previous": "layer1"
        },
        "output": {
            "type": "dense",
            "previous": "layer2"
        }
    },
    "discriminator_layers": {
        "layer1": {
            "type": "dense",
            "previous": "input",
            "size": 256
        },
        "layer2": {
            "type": "relu",
            "previous": "layer1"
        },
        "output": {
            "type": "dense",
            "previous": "layer2",
            "size": 1
        }
    },
    "generator_optimizer": "adam",
    "generator_optimizer_param": {
        "learning_rate": 0.0001,
        "beta1": 0.9,
        "beta2": 0.99
    },
    "discriminator_optimizer": "adam",
    "discriminator_optimizer_param": {
        "learning_rate": 0.0001,
        "beta1": 0.9,
        "beta2": 0.99
    },
    "init": "he_normal",
    "iter": 200000,
    "batch_size": 128,
    "warm_start": 0,
    "num_gen_per_disc": 1,
    "lambd": 10
}

# Parameters for training:

Parameter	Description
init	how the weights should be initialized
iter	total number of iterations for training
batch_size	batch size for each iteration
warm_start_iter	in case if model was saved before which iteration to use to continue training
num_gen_per_disc	how many iterations per row we train generator after training discriminator
lambd	regualarization parameter for discriminator
generator_layers	sequence of generator layers
discriminator_layers	sequence of discriminator layers
generator_optimizer	type of optimizer for generator
discriminator_optimizer	type of optimizer for discriminator
The following Python code creates object of class CDRAGAN and trains GAN:

net = CDRAGAN(header_file, json_file, model_folder, log_file, device)
net.fit(train_file)
Generate new data from the trained GAN model

# To generate new observations from saved GAN model, use the following Python code:

net = CDRAGAN(header_file, json_file, model_folder, log_file, device)
net.generate(train_file, output_file, num_sample_sim, using_iter)

# Postprocess generated data
The GAN model will generate scaled data and if you need to obtain the data in the same format as your original training data, you should apply postprocessing steps. The same workflow as for data preprocessing can be applied. For example, if you applied one-hot encoding, Box-Cox transformation, normalization and missing values treatment, then you should apply the same steps but in reverse order:

mtr = PandasMissingTreatment(json_file=json_file_missing)
generated = mtr.transform(generated)

scl = PandasStandardScaler(json_file=json_file_scale)
generated = scl.back_transform(generated)

bcx = PandasBoxCox(json_file=json_file_boxcox)
generated = bcx.back_transform(generated)

onehot = PandasOneHotEncoder(json_file=json_file_onehot)
generated = onehot.back_transform(generated)

# Additionmal information
Interactive GAN training: https://poloclub.github.io/ganlab/
GAN Zoo: https://github.com/hindupuravinash/the-gan-zoo

# To do
In GAN settings, parameter 'init' is currently not used.
