# Rewriting Fiction
This repository contains the source codes and resources of the Rewriting Fcition project.

## Demo
Please check http://www.fictionrewriting.com:8000/ for the web demo of Rewriting Fiction project. 

## Requirement
- Create an virtual environment
- Python 3.7
- Install the requirements: `pip install -r requirements.txt`
- Download GoogleNews word vector `GoogleNews-vectors-negative300.bin` and save at the root
- `w2v.txt` is our own trained vector. The training process follows https://github.com/mfaruqui/retrofitting 
- apply for a translate api. (I use caiyun here https://open.caiyunapp.com/LingoCloud_API_in_5_minutes)

## Run
You could simply run the `rewrinting.py`and see the result of the rewriting.