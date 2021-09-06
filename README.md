# Rewriting Fiction
This repository contains the source codes and resources of the Rewriting Fiction project.

## Demo
Please check http://www.fictionrewriting.com:8000/ for the web demo of Rewriting Fiction project. 

Check https://docs.google.com/presentation/d/1-Hcsl7yCvYbU-VgITnBw63wB3Clx8KzSDDGKhewTogc/edit?usp=sharing and https://docs.google.com/presentation/d/1u2igm_d9kIaeEIcn-gQGOD7j0dmF5SmyJyhmGyH4ABI/edit?usp=sharing for the slides and poster.
## Requirement
- Create an virtual environment.
- Python 3.7
- Install the requirements: `pip install -r requirements.txt`
- Download GoogleNews word vector `GoogleNews-vectors-negative300.bin` and save at the root
- `w2v.txt` is our own trained vector. The training process follows https://github.com/mfaruqui/retrofitting 
- Apply for a translate api. (I use caiyun here https://open.caiyunapp.com/LingoCloud_API_in_5_minutes)
- Your own text collection.
## Run
You could simply run the `rewrinting.py`and see the result of the rewriting.

## Rewriting Examples
Some more examples for our evaluation and the rewriting results can be found here:
- https://docs.google.com/forms/d/1-mvLP0IrC73fGKOLZ1xwvpFzSeAmPuYZtUvCYQCIfwA/viewform?edit_requested=true
- https://docs.google.com/forms/d/16YRk_SaJbFBNnQZ2Vq_Pln7iG8PHUc-psu7sqB9n60w/viewform?edit_requested=true

## Citation
Please cite our paper if you use `Rewriting Fiction`:
```angular2
@inproceedings{dou2021rf,
  title={Rewriting Fictional Texts using Pivot Paraphrase Generation and Character Modification},
  author={Dou, Liu and Tingting, Zhu and Jörg, Schlötterer and Christin, Seifert and Shenghui, Wang},
  booktitle={International Conference on Text, Speech, and Dialogue},
  year = {2021}
}
```
