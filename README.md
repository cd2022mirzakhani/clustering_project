# CodeUp-DS-Zillow-Zestimate-Project

### Project Goals
* Identify key features that can be used to create an effective predictive model
* Use features to develop a machine learning model that predicts the Zestimate LogErrors

#### Our initial hypothesis is that drivers of LogErrors will be a combination of elements/features that cause an increase of Zestimate LogError.

### Data Dictionary
| Feature | Definition | Type |
|:--------|:-----------|:-------
|**parcelid**|  Unique identifier for parcels (lots) | *int*|
| **bathroooms** |  Number of bathrooms in home |*float*|
| **bedrooms** | Number of bedrooms in home |*int*|
|**sqft**| Area of home in square feet | *int*|
|**year_built**| Year home was built| *int*|
|**latitude**| Latitude of the middle of the parcel | *float*|
|**longitude**| Longitude of the middle of the parcel | *float*|
|**lotsizesquarefeet**|  Area of the lot in square feet | *float*|
|**fireplace**| Is there a fireplace? 1-Yes, 0-No  | *float*|
|**deck**| Is there a deck? 1-Yes, 0-No | *float*|
|**pool**| Is there a pool? 1-Yes, 0-No  | *float*|
|**garage**| Is there a garage? 1-Yes, 0-No  | *float*|
|**hottub**| Is there a hottub? 1-Yes, 0-No  | *float*|
|**county**| County where the home is located | *string*|
|**2017_age**| Age of the home as of 2017 | *int*|
|**Target Variable**
|**logerror**| 𝑙𝑜𝑔𝑒𝑟𝑟𝑜𝑟=𝑙𝑜𝑔(𝑍𝑒𝑠𝑡𝑖𝑚𝑎𝑡𝑒)−𝑙𝑜𝑔(𝑆𝑎𝑙𝑒𝑃𝑟𝑖𝑐𝑒) | *float* |



### The Plan
* Aquire data from the CodeUp database
* Prepare data for exploration by creating tailored columns from existing data

#### Explore data in search of drivers by asking the basic following questions:

* What is the distribution of LogError by county?
* What features should we investigate?

#### Develop a Model to predict LogError

* Use drivers identified to build predictive models of different types
* Evaluate models on train and validate data samples
* Select the best model based on RSME
* Evaluate the best model on test data samples

#### Draw conclusions

### Steps to Reproduce
* Clone this repo.
* Confirm variables from user env.py file as
        username = 'your username', 
        password = 'your password', 
        host = 'data.codeup.com'password pwd, etc.)
* Acquire the data from CodeUp database
* Put the data in the file containing the cloned repo.
* Run notebook
### Conclusions



### Key Takeaway


### Recommendations
