# Recommender System: Identifying Client Needs for a Personal Development School

Project Overview

(Note: This project was created in 2017)

## This project is a recommendation system designed to analyze client interactions and identify:
	1.	Courses that may benefit the client
	2.	The client’s willingness to purchase recommended courses
	3.	The potential budget the client is willing to spend
    
At each stage, the system assigns a probability to its predictions, such as:    
	•	“The client is likely ready to purchase courses.”    
	•	“The client is definitely ready to purchase courses.”    
    
This is the first version of the system.
### For an overview, check out the demo screenshots in the [Demo_Screenshots](https://github.com/shereshevskiy/myprojects/tree/master/personality_development_school/%D0%94%D0%B5%D0%BC%D0%BE_%D1%81%D0%BA%D1%80%D0%B8%D0%BD%D1%88%D0%BE%D1%82%D1%8B) folder
    
## Key Features

	1.	The system’s recommendations are based on analyzing client correspondence and can:
	•	Be displayed on web pages customized for each potential client.
	•	Assist customer support teams during client interactions.
	2.	Demo screenshots of the system’s interface are available in the Demo Screenshots folder.
	3.	The system’s mathematical logic is implemented in Python 3, while the interface page is built using the Bootstrap framework.

## How to Run the Demo

Prerequisites

	•	Operating System: Windows
	•	Python 3 Installed: If Python 3 is not installed, refer to the instructions below for installation guidance.

### Steps to Run

	1.	Clone the Repository    
Download the repository to your computer, ensuring the data folder is included.    
	2.	Run the Application    
Open a command prompt and execute the following command:    
```
python path_to_dir\integra_project_demo.py
```
Replace path_to_dir with the path to the project folder on your computer.

	3.	Wait for Initialization
After running the command, the console will display:

Preparing classifier    
Classifier is ready    
43.610289335250854 seconds    
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)


	4.	Access the Demo
Open your browser and navigate to:
```
http://127.0.0.1:5000/integra-demo
```
The demo interface, as shown in the attached screenshots, should appear.

	5.	Use the System
Input sample client correspondence and click the “Evaluate Client Needs” button to generate recommendations.

### Installing Python 3

If Python 3 is not installed, follow these steps:
	•	Option 1: Download and install Python 3 directly from the official Python website.
	•	Option 2: Install Python as part of the Anaconda distribution:
	•	Visit the Anaconda website.
	•	Download and install the appropriate version for your system.

## Future Improvements

	•	Enhanced AI Models: Incorporate more advanced natural language processing techniques for better accuracy.
	•	Additional Platforms: Extend compatibility to Linux and MacOS.
	•	Cloud Deployment: Deploy the system on a cloud platform for easier access without local setup.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to share feedback or suggestions to improve the system! 😊
