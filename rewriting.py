from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import stanza as st
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
import csv
import textstat
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from gensim.models import KeyedVectors
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import tokenize
from nltk.corpus import wordnet

surname = np.load('surname.npy', allow_pickle=True)
malenames = np.load('male.npy', allow_pickle=True)
femalenames = np.load('female.npy', allow_pickle=True)
unsex = np.load('unsex.npy', allow_pickle=True)
boy = np.load('boy.npy', allow_pickle=True)
girl  = np.load('girl.npy', allow_pickle=True)
gender_word=[['boy', 'girl'], ['boys', 'girls'], ['nephew', 'niece'], ['brother', 'sister'], ['brothers', 'sisters'],['boyfriend', 'girlfriend'],
             ['dad', 'mom'], ['father', 'mother'], ['grandfather', 'grandmother'], ['grandpa', 'grandma'], ['grandson', 'granddaughter'], ['male','female'],
             ['groom', 'bride'], ['husband', 'wife'], ['king', 'queen'], ['man', 'woman'],['men','women'], ['policeman', 'policewoman'], ['prince', 'princess'],
             ['son', 'daughter'], ['sons', 'daughters'], ['stepbrother', 'stepsister'], ['stepfather', 'stepmother'], ['stepson', 'stepdaughter'],['enchanter','enchantress'],
             ['uncle', 'aunt'], ['host', 'hostess'], ['landlord', 'landlady'], ['waiter', 'waitress'], ['emperor', 'empress'], ['steward', 'stewardess'],['witch','wizard'],
             ['Boy', 'Girl'], ['Boys', 'Girls'], ['Nephew', 'Niece'], ['Brother', 'Sister'], ['Brothers', 'Sisters'],['Boyfriend', 'Girlfriend'],
             ['Dad', 'Mom'], ['Father', 'Mother'], ['Grandfather', 'Grandmother'], ['Grandpa', 'Grandma'], ['Grandson', 'Granddaughter'],   ['Male','Female'],
             ['Groom', 'Bride'], ['Husband', 'Wife'], ['King', 'Queen'], ['Man', 'Woman'], ['Policeman', 'Policewoman'], ['Prince', 'Princess'],
             ['Son', 'Daughter'], ['Sons', 'Daughters'], ['Stepbrother', 'Stepsister'], ['Stepfather', 'Stepmother'], ['Stepson', 'Stepdaughter'],
             ['Uncle', 'Aunt'], ['Host', 'Hostess'], ['Landlord', 'Landlady'], ['Waiter', 'Waitress'], ['Emperor', 'Empress'], ['Steward', 'Stewardess']]
male = ['he', 'him', 'his', 'himself']
female = ['her', 'she', 'herself']
pronons = ['her', 'she', 'herself', 'he', 'him', 'his', 'himself', 'I', 'my', 'me', 'mine', 'they', 'their', 'them']


# change the gender of pronouns
# 目前是改变所有的性别代词
# 考虑是否只改变部分角色的性别
'''
change the gender of pronouns
input: stanza document
return: sentence after detoken
目前是改变所有的性别代词
考虑是否只改变部分角色的性别
'''
def change_gender(doc):
    words=[]
    for sentence in doc.sentences:

        for word in sentence.words:
            if word.text == 'her' and (word.deprel == 'obj' or word.deprel == 'nmod' or word.deprel == 'nsubj'):
                word.text = 'him'
            elif word.text == 'her':
                word.text = 'his'
            elif word.text == 'she':
                word.text = 'he'
            elif word.text == 'Her':
                word.text = 'His'
            elif word.text == 'She':
                word.text = 'He'
            elif word.text == 'herself':
                word.text = 'himself'
            elif word.text == 'he':
                word.text = 'she'
            elif word.text == 'him' or word.text =='his':
                word.text = 'her'
            elif word.text == 'himself':
                word.text = 'herself'
            elif word.text == 'He':
                word.text = 'She'
            elif word.text == 'His':
                word.text = 'Her'
            words.append(word.text)

    # detoken, compose the tokens to sentence
    detoken = TreebankWordDetokenizer().detokenize(words)
    return detoken

'''old function for change gender

def replace_gender(text):

    temp = text.replace(" his "," hiss ").replace(" he "," hee ").replace(" him "," himm ").replace(" His "," Hiss ").replace(" He "," Hee ").replace(" Him "," Himm ").replace(" himself "," himselff ")
    docgender = en_nlp(temp)
    sent = []
    for sentence in docgender.sentences:
        words = []
        for word in sentence.words:
            if word.text == 'her' and (word.deprel == 'obj' or word.deprel == 'nmod' or word.deprel == 'nsubj'):
                word.text = 'him'
            elif word.text == 'her':
                word.text = 'his'
            elif word.text == 'she':
                word.text = 'he'
            elif word.text == 'Her':
                word.text = 'His'
            elif word.text == 'She':
                word.text = 'He'
            elif word.text == 'herself':
                word.text = 'himself'
            words.append(word.text)
        sent.append(" ".join(words))
    text_new = " ".join(sent)
    text_new = text_new.replace(" hiss "," her ").replace(" hee "," she ").replace(" himm "," her ").replace(" Hiss "," Her ").replace(" Hee "," She ").replace(" Himm "," Her ").replace(' himselff '," herself ")
    return text_new
'''

'''
get all the names with the probility of gender
'''
def get_allname():
    csv_file = csv.reader(open('C:\\Users\\LDLuc\\Downloads\\2020-04\\REDI\\name_gender.csv', 'r'))
    male = []
    female = []
    unsex = []
    for x in csv_file:
        if x[2] == 'probability':
            pass
        elif x[1] == 'M' and float(x[2]) > 0.5:
            male.append(x[0])
        elif x[1] == 'F' and float(x[2]) > 0.5:
            female.append(x[0])
        elif float(x[2]) == 0.5:
            unsex.append(x[0])
    return male,female,unsex


'''
'''
def get_role(doc):
    entity=[]
    result_stan = doc.entities
    for x in result_stan:
        if x.type == "PERSON":
            entity.append(x.text)
    role = list(set(entity))
    # print(role)
    delet=[]
    for x in role:
        for y in role:
            if x!=y and x in y:
                if x in y.split():
                    delet.append(x)
    # delet = list(set(delet))
    print("delet is ",delet)
    for y in delet:
        if y in role:
            role.remove(y)
    print("role is", role)
    return role,delet

def change_rolename(longtext,role,delet):
    originalnames = []
    name_pair = []
    newnames= []
    new_role = copy.deepcopy(role)
    for x in role:
        name_list = []
        if len(x.split()) == 2:

            name_list.append(x.split()[0])
            name_list.append(x.split()[1])
            for y in delet:
                if y in x.split():
                    if y.capitalize() in malenames:
                        number1 = random.randint(0,len(girl)-1)
                        number2 = random.randint(0,len(surname)-1)
                        name_list.append(girl[number1])
                        name_list.append(surname[number2])
                        print(name_list)
                        originalname = name_list[0] + ' ' + name_list[1]
                        newname = name_list[2] + ' ' + name_list[3]
                        originalnames.append(originalname)
                        newnames.append(newname)
                        if x in new_role:
                            new_role.remove(x)
                        longtext = longtext.replace(originalname, newname)
                    elif y.capitalize() in femalenames:
                        number1 = random.randint(0, len(boy) - 1)
                        number2 = random.randint(0, len(surname) - 1)
                        name_list.append(boy[number1])
                        name_list.append(surname[number2])
                        print(name_list)
                        originalname = name_list[0] + ' ' + name_list[1]
                        newname = name_list[2] + ' ' + name_list[3]
                        originalnames.append(originalname)
                        newnames.append(newname)
                        if x in new_role:
                            new_role.remove(x)
                        longtext = longtext.replace(originalname, newname)
    print(new_role)

    for y in delet:
        for z1,z2 in zip(originalnames,newnames):
            if y in z1:
                firstname = z2.split()[0]
                token = tokenize.word_tokenize(longtext)
                for i, k in enumerate(token):
                    if k == y:
                        token[i] = firstname
                longtext = TreebankWordDetokenizer().detokenize(token)

    for rest in new_role:
        if len(rest.split()) == 1:
            if rest in malenames:
                number = random.randint(0, len(girl)-1)
                longtext = longtext.replace(rest, girl[number])
            elif rest in femalenames:
                number = random.randint(0, len(boy)-1)
                longtext = longtext.replace(rest, boy[number])
            elif rest in unsex:
                number = random.randint(0, len(unsex)-1)
                longtext = longtext.replace(x, unsex[number])
        elif len(rest.split()) > 1:

            for part in rest.split():
                if part in malenames:
                    number1 = random.randint(0, len(girl) - 1)
                    number2 = random.randint(0, len(surname) - 1)
                    newname = girl[number1] + ' ' + surname[number2]
                    longtext = longtext.replace(rest, newname)
                    break
                elif part in femalenames:
                    number1 = random.randint(0, len(boy) - 1)
                    number2 = random.randint(0, len(surname) - 1)
                    newname = boy[number1] + ' '+surname[number2]
                    longtext = longtext.replace(rest, newname)
                    break
                elif part in unsex:
                    number1 = random.randint(0, len(boy) - 1)
                    number2 = random.randint(0, len(surname) - 1)
                    newname = boy[number1] + ' ' + surname[number2]
                    longtext = longtext.replace(rest, newname)
                    break
    print(longtext)
    print(new_role)
    return longtext

def change_genderwords(text):
    token = tokenize.word_tokenize(text)
    for count, tokens in enumerate(token):
        for words in gender_word:
            if words[0] == tokens:
                token[count] = words[1]
            elif words[1] == tokens:
                token[count] = words[0]
    detoken = TreebankWordDetokenizer().detokenize(token)
    return detoken

def find_replace_adj(text,doc,model):
    adj = []
    new_adj = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.text not in adj and (word.upos == 'ADJ' or word.xpos == 'JJ'):
                adj.append(word.text)
    token = tokenize.word_tokenize(text)
    for count, tokens in enumerate(token):
        for words in adj:
            if words == tokens:
                if words in model.vocab:
                    token[count] = model.most_similar(positive=[words])[0][0]
                    new_adj.append(token[count])
                else:
                    pass
    detoken = TreebankWordDetokenizer().detokenize(token).replace(' .', '.')
    return detoken,adj,new_adj
new_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

# rePlace with GoogleNews Vectors
def find_replace_adj_new(text,doc):
    # new_model = KeyedVectors.load_word2vec_format('venv\\out_vec.txt')
    adj = []
    new_adj = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.text not in adj and (word.upos == 'ADJ' or word.xpos == 'JJ'):
                adj.append(word.text)
    token = tokenize.word_tokenize(text)
    for count, tokens in enumerate(token):
        for words in adj:
            if words == tokens:
                if words in new_model.vocab:
                    token[count] = new_model.most_similar(positive=[words])[0][0]
                    new_adj.append(token[count])
                else:
                    pass
    detoken = TreebankWordDetokenizer().detokenize(token).replace(' .', '.').replace(' - ','-')
    return detoken,adj,new_adj

def wordnet_replace(text,doc,mode='synonyms'):
    adj = []
    new_adj = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.text not in adj and (word.upos == 'ADJ' or word.xpos == 'JJ'):
                adj.append(word.text)

    token = tokenize.word_tokenize(text)

    for x in adj:
        synonyms = []
        antonyms = []

        for syn in wordnet.synsets(x):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
        synonyms=list(set(synonyms))
        antonyms=list(set(antonyms))
        if mode=='synonyms':
            if len(synonyms) ==1:
                for count, tokens in enumerate(token):
                    if tokens == x:
                        token[count]=synonyms[0]
            elif len(synonyms) >1:
                # number = random.randint(0, len(synonyms))
                for count,tokens in enumerate(token):
                    number = random.randint(0,len(synonyms)-1) # 每次发现adj里的单词，随机替换一个同义词
                    if tokens == x:
                        # print(synonyms)
                        # print(number)
                        token[count] = synonyms[number]
        elif mode=='antonyms':
            if len(antonyms)==1:
                for count,tokens in enumerate(token):
                    if tokens ==x:
                        token[count] = antonyms[0]
            elif len(antonyms)>1:
                for count,tokens in enumerate(token):
                    number = random.randint(0,len(antonyms)-1)
                    if tokens == x:
                        token[count]=antonyms[number]
    detoken = TreebankWordDetokenizer().detokenize(token).replace(' .', '.')
    return detoken

def tranlate(source, direction):
    import requests
    import json

    url = "http://api.interpreter.caiyunai.com/v1/translator"

    # WARNING, this token is a test token for new developers, and it should be replaced by your token
    token = "gmqakykfmcb5x4n7pty7"

    payload = {
        "source": source,
        "trans_type": direction,
        "request_id": "demo",
        "detect": True,
    }

    headers = {
        'content-type': "application/json",
        'x-authorization': "token " + token,
    }

    response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
    return json.loads(response.text)['target']



from gensim.models import KeyedVectors

en_nlp = st.Pipeline('en')
# our own model + retrofitting with semantic lexicon
model = KeyedVectors.load_word2vec_format('retrofitting\\w2v.txt')



def test_one(longtext):

    longtext = tranlate(tranlate([longtext], "auto2zh"),'zh2en')
    doc2 = en_nlp(longtext)
    gender_change = change_gender(doc2)
    role, delet = get_role(doc2)
    rolechange = change_rolename(gender_change, role, delet)
    genderwords = change_genderwords(rolechange)
    adjchange, adj, new_adj = find_replace_adj(genderwords, doc2, model)
    adjchange_new, adj_2, new_adj_2 = find_replace_adj_new(genderwords,doc2)
    print(adjchange,'\n original adj',adj,'\n original new',new_adj)
    print(adjchange_new,'\n original',adj_2,'\n retrofitting',new_adj_2)

    wordnet_sent2 = wordnet_replace(genderwords, doc2)
    print("original : ",longtext)
    print("w2v : ",adjchange)
    print("wordnet : ",wordnet_sent2)
    print("google : ",adjchange_new)
    return adjchange,wordnet_sent2, adjchange_new,longtext[0]


def test(y):
    m=0
    trans_wordnet = []
    trans_w2v = []
    trans_w2v_google = []
    # longtext="One woman is about to discover everything she believes--knows--to be true about her life ... isn't. After hitting her head, Lucy Sparks awakens in the hospital to a shocking revelation: the man she's known and loved for years--the man she recently married--is not actually her husband. In fact, they haven't even spoken since their breakup four years earlier. The happily-ever-after she remembers in vivid detail--right down to the dress she wore to their wedding--is only one example of what her doctors call a false memory: recollections Lucy's mind made up to fill in the blanks from the coma. Her psychologist explains the condition as honest lying, because while Lucy's memories are false, they still feel incredibly real. Now she has no idea which memories she can trust--a devastating experience not only for Lucy, but also for her family, friends and especially her devoted boyfriend, Matt, whom Lucy remembers merely as a work colleague. When the life Lucy believes she had slams against the reality she's been living for the past four years, she must make a difficult choice about which life she wants to lead, and who she really is."
    df = np.load('plot_fiction.npy',allow_pickle=True)
    for x in df[:y]:
        m+=1
        x = tranlate(tranlate([x], "auto2zh"), 'zh2en')
        doc2 = en_nlp(x[0])
        gender_change = change_gender(doc2)
        role,delet =get_role(doc2)
        rolechange = change_rolename(gender_change,role,delet)
        genderwords = change_genderwords(rolechange)
        adjchange,adj,new_adj = find_replace_adj(genderwords,doc2,model)
        adjchange_new, adj_2, new_adj_2 = find_replace_adj_new(genderwords, doc2)

        wordnet_sent2 = wordnet_replace(genderwords,doc2)
        trans_w2v_google.append(adjchange_new)
        trans_wordnet.append(wordnet_sent2)
        trans_w2v.append(adjchange)
        print('now is  %d'% m)
    return df[:y],trans_w2v,trans_wordnet,trans_w2v_google


def readability(plot_1,plot_2):
    flesch = np.zeros(4)
    flesch_n = np.zeros(4)
    gun = np.zeros(4)
    cole_n = np.zeros(4)
    dale = np.zeros(4)
    gun_n = np.zeros(4)
    cole = np.zeros(4)
    dale_n = np.zeros(4)
    for plot1,plot2 in zip(plot_1,plot_2):
        flesch1 = textstat.flesch_reading_ease(plot1)
        flesch2 = textstat.flesch_reading_ease(plot2)

        if abs(flesch1-flesch2) <= 5:
            if flesch1>flesch2: # flesch1 is easier
                flesch[0] -=1
            else:
                flesch_n[0]+=1
        elif abs(flesch1-flesch2) > 5 and abs(flesch1-flesch2) <= 10:
            if flesch1>flesch2: # flesch1 is easier
                flesch[1] -=1
            else:
                flesch_n[1]+=1
        elif abs(flesch1-flesch2) > 10 and abs(flesch1-flesch2) <= 15:
            if flesch1>flesch2: # flesch1 is easier
                flesch[2] -=1
            else:
                flesch_n[2]+=1
        elif abs(flesch1-flesch2) > 15:
            if flesch1>flesch2: # flesch1 is easier
                flesch[3] -=1
            else:
                flesch_n[3]+=1
        gun1 = textstat.gunning_fog(plot1)
        gun2 = textstat.gunning_fog(plot2)
        if abs(gun1 - gun2) <= 0.5:
            if gun1 > gun2: # gun1 is harder
                gun_n[0] +=1
            else:
                gun[0] -= 1

        elif abs(gun1 - gun2) > 0.5 and abs(gun1 - gun2) <= 1.0:
            if gun1 > gun2:  # gun1 is harder
                gun_n[1] += 1
            else:
                gun[1] -= 1
        elif abs(gun1 - gun2) > 1.0 and abs(gun1 - gun2) <= 1.5:
            if gun1 > gun2:  # gun1 is harder
                gun_n[2] += 1
            else:
                gun[2] -= 1
        elif abs(gun1 - gun2) > 1.5:
            if gun1 > gun2:  # gun1 is harder
                gun_n[3] += 1
            else:
                gun[3] -= 1
        cole1 = textstat.automated_readability_index(plot1)
        cole2 = textstat.automated_readability_index(plot2)
        if abs(cole1 - cole2) <= 1:
            if cole1 > cole2:  #  cole1 is harder
                cole_n[0] += 1
            else:
                cole[0] -= 1
        elif abs(cole1 - cole2) > 1 and abs(cole1 - cole2) <= 2:
            if cole1 > cole2:  #  cole1 is harder
                cole_n[1] += 1
            else:
                cole[1] -= 1
        elif abs(cole1 - cole2) > 2 and abs(cole1 - cole2) <= 3:
            if cole1 > cole2:  #  cole1 is harder
                cole_n[2] += 1
            else:
                cole[2] -= 1
        elif abs(cole1 - cole2) > 3:
            if cole1 > cole2:  #  cole1 is harder
                cole_n[3] += 1
            else:
                cole[3] -= 1
        dale1 = textstat.dale_chall_readability_score(plot1)
        dale2 = textstat.dale_chall_readability_score(plot2)
        if abs(dale1 - dale2) <= 0.5:
            if dale1 > dale2:  #  cole1 is harder
                dale_n[0] += 1
            else:
                dale[0] -= 1
        elif abs(dale1 - dale2) > 0.5 and abs(dale1 - dale2) <= 1.0:
            if dale1 > dale2:  #  cole1 is harder
                dale_n[1] += 1
            else:
                dale[1] -= 1
        elif abs(dale1 - dale2) > 1.0 and abs(dale1 - dale2) <= 1.5:
            if dale1 > dale2:  #  cole1 is harder
                dale_n[2] += 1
            else:
                dale[2] -= 1
        elif abs(dale1 - dale2) > 1.5:
            if dale1 > dale2:  #  cole1 is harder
                dale_n[3] += 1
            else:
                dale[3] -= 1
    average = []
    average_n = []
    for i in range(4):
        average.append((flesch[i]+gun[i]+cole[i]+dale[i])/4)
        average_n.append((flesch_n[i]+gun_n[i]+cole_n[i]+dale_n[i])/4)
    x = [1,7,13,19]
    x2 = [i+1 for i in x]
    x3 = [i+2 for i in x]
    x4 = [i+3 for i in x]
    x5 = [i+4 for i in x]
    tick = [i+2 for i in x]
    bar_width = 0.8
    tick_label = ['same level','one level','two levels','more than two levels']
    plt.bar(x,flesch,bar_width,align='center',label = 'flesch',alpha = 0.5,color="blue")
    plt.bar(x2,gun,bar_width,align='center',label = 'gunning_fog',alpha = 0.5,color="#7EE680")
    plt.bar(x3, cole, bar_width, align='center', label='ARI', alpha=0.5,color="#87CEFA")
    plt.bar(x4, dale, bar_width, align='center', label='dale_chall', alpha=0.5,color="red")
    plt.bar(x5, average, bar_width, align='center', label='average', alpha=0.5,color="orange")
    plt.bar(x,flesch_n,bar_width,align='center',color="blue",alpha = 0.5)
    plt.bar(x2,gun_n,bar_width,align='center',alpha = 0.5,color="#7EE680")
    plt.bar(x3, cole_n, bar_width, align='center', alpha=0.5,color="#87CEFA")
    plt.bar(x4, dale_n, bar_width, align='center', alpha=0.5,color="red")
    plt.bar(x5, average_n, bar_width, align='center', alpha=0.5,color="orange")

    plt.xlabel('Differences in the level of education required to understand texts')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks(tick, tick_label)
    plt.title('Differences on Readability between Original and Rewriting Fiction')
    plt.show()

text = ["The happily-ever-after she remembers in vivid detail--right down to the dress she wore to their wedding--is only one example of what her doctors call a false memory: recollections Lucy's mind made up to fill in the blanks from the coma.",
        "Highland warrior Ewan Brody always wanted a sweet, uncomplicated woman by his side, but he can't fight his attraction to the beautiful enchantress who's stumbled into his life. ",
        "Tre is determined to let no one stand in his way, not even the captivating Lady Jane Neville, a known sympathizer to the Saxon cause whose unbridled spirit evokes feelings in Tre he thought were long buried.",
        "Amelia Ann Hollins might have been raised to be a sweet magnolia, but she's found her inner fire--the quest for justice. ",
        "She soon finds that she's living in an area by-passed by time and progress, as she butts heads with stubborn old ranchers who want nothing to do with a young female vet. ",
        "Once known as the high school weirdo, Camden is bigger and badder than the boy he used to be and a talented tattoo artist with his own thriving business.",
        "Meanwhile, several young children from the nearby Indian reservation have gone missing, and Beth fears that something is pursuing her in the bush. ",
        "Then in walks Luther Beale, the notorious vigilante who five years ago shot a boy for vandalizing his car.",
        "But she arrives home to find that her estranged mother - renowned archaeologist Jennifer Almieri - is dead, and the investigation into her death is being handled by Starfleet. ",
        "Liam Kelly is many things: a former wheelman for the IRA, a one-time political prisoner, the half-breed son of a mystic Fey warrior and a mortal woman, and a troubled young man literally haunted by the ghosts of his past."]
