import stanza as st
import csv
import textstat
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from gensim.models import KeyedVectors
from senticnet.senticnet import SenticNet
sn = SenticNet()

import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

from transformers import pipeline
from wordhoard import antonyms,synonyms
from nltk import tokenize
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer
nlp = pipeline("fill-mask",model="bert-large-uncased",tokenizer="bert-large-uncased")


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

'''
change the gender of pronouns
input: stanza document
return: sentence after detoken
currently, we change all the pronouns exist in the text.
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


'''
get all the names with the probility of gender
'''
def get_allname():
    csv_file = csv.reader(open('name_gender.csv', 'r'))
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
to get the roles' list in a fiction which contains all roles appears more than once
input: the Stanza doc.
output: a role's list and a delet list
delete list contains the role name which is not the whole name, for example, role lis= ['Lucy White'] and the delet list = ['Lucy']
In fiction, the name is presented as the full name when the first time a role is introduced and then the author will refer this role as his/her first name
Thus we should know that these two name should belong to the same person (this can also be implemented using Coreference Resolution, a complex way)
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

'''
input: input text, role list & delet list
output: new text change the role's name to another name with different gender

Example: Bob Dylan --> Alice White  Bob --> Alice

'''
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

'''
input: text need to change some words which are gender sensetive
output: text changed

For exanple: king --> queen or father's --> mother's

'''
def change_genderwords(text):
    newtext = ""
    for sentence in tokenize.sent_tokenize(text):
        token = tokenize.word_tokenize(sentence)
        for count, tokens in enumerate(token):
            for words in gender_word:
                if words[0] == tokens:
                    token[count] = words[1]
                elif words[1] == tokens:
                    token[count] = words[0]
                elif words[0] + '\'s' == tokens:
                    token[count] = words[1] + '\'s'
                elif words[1] + '\'s' == tokens:
                    token[count] = words[0] + '\'s'
        detoken = TreebankWordDetokenizer().detokenize(token)
        newtext = newtext + detoken + ' '
    return newtext

def find_replace_adj(text,doc,model,mode='antonym'):
    adj = []
    new_adj = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.text not in adj and (word.upos == 'ADJ' or word.xpos == 'JJ'):
                adj.append(word.text)
    token = tokenize.word_tokenize(text)
    if mode == 'antonym':
        for count, tokens in enumerate(token):
            for words in adj:
                if words == tokens:
                    if words in model.vocab:
                        try:
                            polarity_intense = sn.polarity_value(words)
                            if float(polarity_intense) > 0.:
                                token[count] = \
                                model.most_similar(positive=['negative', words], negative=['positive'])[0][0]
                            else:
                                token[count] = \
                                model.most_similar(positive=['positive', words], negative=['negative'])[0][0]
                            # token[count] = model.most_similar(positive=[words])[0][0]
                            new_adj.append(token[count])
                        except:
                            pass

                    else:
                        pass
    elif mode == 'synonym':
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

# rePlace with GoogleNews Vectors
def find_replace_adj_new(text,doc,new_model= KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True),mode='antonym'):
    adj = []
    new_adj = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.text not in adj and (word.upos == 'ADJ' or word.xpos == 'JJ'):
                adj.append(word.text)
    token = tokenize.word_tokenize(text)
    if mode == 'antonym':
        for count, tokens in enumerate(token):
            for words in adj:
                if words == tokens:
                    if words in new_model.vocab:
                        try:
                            polarity_intense = sn.polarity_value(words)
                            if float(polarity_intense) > 0.:
                                token[count] = \
                                new_model.most_similar(positive=['negative', words], negative=['positive'])[0][0]
                            else:
                                token[count] = \
                                new_model.most_similar(positive=['positive', words], negative=['negative'])[0][0]
                            # token[count] = model.most_similar(positive=[words])[0][0]
                            new_adj.append(token[count])
                        except:
                            pass

                    else:
                        pass
    elif mode == 'synonym':
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

def find_synonyms(word):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return list(set(synonyms))
    # return synonyms.query_synonym_com(word)

def find_antonyms(word):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return list(set(antonyms))
    # return antonyms.query_thesaurus_com(word)

def judge_adj(sentence,anto_list,original_word,mode):
    # print(anto_list,original_word)
    if len(original_word) == 0:
        return sentence
    else:
        if mode == 'antonym':
            if len(anto_list) == len(original_word):
                # print(anto_list)
                for index, adj_to_change in enumerate( original_word):
                    # print(index)
                    # print(anto_list[index])
                    if anto_list[index] != '' and anto_list[index]!= []:
                        sentence = sentence.replace(adj_to_change, '{}',1)
                        result = nlp(sentence.format(nlp.tokenizer.mask_token), targets=anto_list[index])
                        new_adj = result[0]['token_str']
                        # sentence = sentence.replace(adj_to_change, '{}')
                        sentence = sentence.replace('{}', new_adj).replace(' ,',',')
                        sentence = sentence[0].upper() + sentence[1:]
                        # print("the sentence is: \n {}".format(sentence))
                return sentence

        elif mode == 'synonym':
            if len(anto_list) == len(original_word):
                for index, adj_to_change in enumerate( original_word):
                    # print(index)
                    # print(anto_list[index])
                    if anto_list[index] != '' and anto_list[index]!= []:

                        sentence = sentence.replace(adj_to_change, '{}',1)
                        # print(sentence)
                        result = nlp(sentence.format(nlp.tokenizer.mask_token))
                        new_adj = result[0]['token_str']
                        # sentence = sentence.replace(adj_to_change, '{}')
                        sentence = sentence.replace('{}', new_adj).replace(' ,',',')
                        sentence = sentence[0].upper()+sentence[1:]
                    # print("the sentence is: \n {}".format(sentence))
                    return sentence

def wordnet_replace_bert(doc,mode='antonym'):
    adj = []
    detoken = ''
    sent_count = 0
    if mode == 'antonym':
        for sent in doc.sentences:
            token_count = 0
            word_find_list = []
            antonym_list = []
            for word in sent.words:
                if word.text not in adj and (word.upos == 'ADJ' or word.xpos == 'JJ'):
                    adj.append((sent_count,token_count ,word.text))
                    word_find_list.append(word.text)
                    antonym = find_antonyms(word.text)

                    antonym_list.append(antonym)
                # token_count +=1
                # detoken.append(word.text)
            sentence = judge_adj(sent.text,antonym_list,word_find_list,mode)
            detoken = detoken + sentence +' '
    elif mode == 'synonym':
        for sent in doc.sentences:
            token_count = 0
            word_find_list = []
            antonym_list = []
            for word in sent.words:
                if word.text not in adj and (word.upos == 'ADJ' or word.xpos == 'JJ'):
                    adj.append((sent_count,token_count ,word.text))
                    word_find_list.append(word.text)
                    antonym = find_synonyms(word.text)
                    antonym_list.append(antonym)
                # token_count +=1
                # detoken.append(word.text)
            sentence = judge_adj(sent.text,antonym_list,word_find_list,mode)
            detoken = detoken + sentence +' '

    print(detoken)
    return detoken

def wordnet_replace(text,doc,mode='synonym'):
    adj = []
    new_adj = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.text not in adj and (word.upos == 'ADJ' or word.xpos == 'JJ'):
                adj.append(word.text)

    token = tokenize.word_tokenize(text)

    for x in adj:
        synonyms = find_synonyms(x)
        antonyms = find_antonyms(x)

    
        if mode=='synonym':
            try:
                if len(synonyms) ==1:
                    for count, tokens in enumerate(token):
                        if tokens == x:
                            token[count]=synonyms[0]
                elif len(synonyms) >1:
                    # number = random.randint(0, len(synonyms))
                    for count,tokens in enumerate(token):
                        number = random.randint(0,len(synonyms)-1) # 每次发现adj里的单词，随机替换一个同义词
                        if tokens == x:
                            
                            token[count] = synonyms[number]
            except:
                print("{} has no synonym in wordhard".format(x))
        elif mode=='antonym':
            print("the adj is :{}".format(x))
            print("the antonyms is :{}".format(antonyms))
            try:
                if len(antonyms)==1:
                    for count,tokens in enumerate(token):
                        if tokens ==x:
                            token[count] = antonyms[0]
                elif len(antonyms)>1:
                    for count,tokens in enumerate(token):
                        number = random.randint(0,len(antonyms)-1)
                        if tokens == x:
                            token[count]=antonyms[number]
            except:
                print("{} has no antonyms in wordhard".format(x))

    detoken = TreebankWordDetokenizer().detokenize(token).replace(' .', '.')
    return detoken

def tranlate(source, direction):
    import requests
    import json

    url = "http://api.interpreter.caiyunai.com/v1/translator"

    # WARNING, this token is a test token for new developers, and it should be replaced by your token
    token = "4oxzhu5tic0kd4xefpu0"

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




def test_one(longtext):
    print('original:',longtext)
    matches_original = tool.check(longtext)

    longtext = tranlate(tranlate([longtext], "auto2zh"),'zh2en')[0]
    longtext = longtext.replace('’','\'')
    print('paraphrase',longtext)
    matches_paraphrase = tool.check(longtext)

    doc2 = en_nlp(longtext)
    gender_change = change_gender(doc2)
    role, delet = get_role(doc2)
    rolechange = change_rolename(gender_change, role, delet)
    genderwords = change_genderwords(rolechange)
    doc3 = en_nlp(genderwords)

    # use own vector to find similar
    adjchange, adj, new_adj = find_replace_adj(genderwords, doc2, model,mode='synonym')
    matches_own_vec_syn = tool.check(adjchange)
    # use google vector to find similar
    adjchange_new, adj_2, new_adj_2 = find_replace_adj_new(genderwords,doc2,mode='synonym')
    matches_google_vec_syn  = tool.check(adjchange_new)
    # use word_net to find synomys
    wordnet_sent2 = wordnet_replace(genderwords, doc2,mode='synonym')
    matches_wordnet_syn  = tool.check(wordnet_sent2)

    bert_synonyms = wordnet_replace_bert(doc3,mode='synonym')
    matches_bert_syn  = tool.check(bert_synonyms)

    adjchange_antonym, adj_antonym, new_adj_antonym = find_replace_adj(genderwords, doc2, model, mode='antonym')
    adjchange_new_antonym, adj_2_antonym, new_adj_2_antonym = find_replace_adj_new(genderwords, doc2, mode='antonym')
    wordnet_sent2_antonym = wordnet_replace(genderwords, doc2, mode='antonym')
    bert_antonym = wordnet_replace_bert(doc3, mode='antonym')
    matches_own_vec_anto = tool.check(adjchange_antonym)
    matches_google_vec_anto = tool.check(adjchange_new_antonym)
    matches_wordnet_anto = tool.check(wordnet_sent2_antonym)
    matches_bert_anto = tool.check(bert_antonym)



    print(adjchange, '\n original adj', adj, '\n retrofitting', new_adj)
    print(adjchange_new, '\n original', adj_2, '\n goole_news', new_adj_2)

    print("original : ",longtext)
    print("w2v : ",adjchange)
    print("wordnet : ",wordnet_sent2)
    print("google : ",adjchange_new)
    print("bert : ",bert_synonyms)
    print("=============================antonyms======================")
    print(adjchange, '\n original adj', adj_antonym, '\n retrofitting', new_adj_antonym)
    print(adjchange_new, '\n original', adj_2_antonym, '\n goole_news', new_adj_2_antonym)
    print("w2v antonym: ", adjchange_antonym)
    print("wordnet antonym: ", wordnet_sent2_antonym)
    print("google antonym: ", adjchange_new_antonym)
    print("bert antonym: ", bert_antonym)

    return matches_original,matches_paraphrase,matches_own_vec_syn,matches_own_vec_anto,matches_google_vec_syn,\
    matches_google_vec_anto,matches_wordnet_syn,matches_wordnet_anto,matches_bert_syn,matches_bert_anto

    # return adjchange,wordnet_sent2, adjchange_new,longtext[0]
    # return adjchange_new,wordnet_sent2, longtext[0]
    # return wordnet_sent2, longtext

from collections import Counter
def statistic(matches_list):
    '''
    input the list of the spell check results, and output the counter result
    '''
    category_list = []
    ruleuid_list = []
    issue_list = []
    for matches in matches_list:
        for rule in matches:
            issue_list.append(rule.ruleIssueType)
            category_list.append(rule.category)
            ruleuid_list.append(rule.ruleId)
    print(Counter(issue_list))
    print(Counter(category_list))
    print(Counter(ruleuid_list))


def test_spellcheck(plot_list):

    matches_original_list = []
    matches_paraphrase_list = []
    matches_own_vec_syn_list = []
    matches_own_vec_anto_list = []
    matches_google_vec_syn_list = []

    matches_google_vec_anto_list = []
    matches_wordnet_syn_list = []
    matches_wordnet_anto_list = []
    matches_bert_syn_list = []
    matches_bert_anto_list = []
    candidates = [matches_original_list, matches_paraphrase_list, matches_own_vec_syn_list, matches_own_vec_anto_list,
                  matches_google_vec_syn_list, \
                  matches_google_vec_anto_list, matches_wordnet_syn_list, matches_wordnet_anto_list,
                  matches_bert_syn_list, matches_bert_anto_list]
    for plot in plot_list:
        matches_original, matches_paraphrase, matches_own_vec_syn, matches_own_vec_anto, matches_google_vec_syn, \
        matches_google_vec_anto, matches_wordnet_syn, matches_wordnet_anto, matches_bert_syn, matches_bert_anto = test_one(plot)
        matches_original_list.append(matches_original)
        matches_paraphrase_list.append(matches_paraphrase)
        matches_own_vec_syn_list.append(matches_own_vec_syn)
        matches_own_vec_anto_list.append(matches_own_vec_anto)
        matches_google_vec_syn_list.append(matches_google_vec_syn)
        matches_google_vec_anto_list.append(matches_google_vec_anto)
        matches_wordnet_syn_list.append(matches_wordnet_syn)
        matches_wordnet_anto_list.append(matches_wordnet_anto)
        matches_bert_syn_list.append(matches_bert_syn)
        matches_bert_anto_list.append(matches_bert_anto)
    num = 1
    for i in candidates:

        print("="*30)
        print(num)
        statistic(i)
        num+=1
    return matches_original_list, matches_paraphrase_list, matches_own_vec_syn_list, matches_own_vec_anto_list,matches_google_vec_syn_list, \
                  matches_google_vec_anto_list, matches_wordnet_syn_list, matches_wordnet_anto_list,matches_bert_syn_list, matches_bert_anto_list



def test(plot):
    m=0
    para = []
    trans_wordnet = []
    trans_w2v = []
    trans_w2v_google = []
    trans_bert = []
    trans_wordnet_anto = []
    trans_w2v_ant = []
    trans_w2v_google_ant = []
    trans_bert_ant = []
    # longtext="One woman is about to discover everything she believes--knows--to be true about her life ... isn't. After hitting her head, Lucy Sparks awakens in the hospital to a shocking revelation: the man she's known and loved for years--the man she recently married--is not actually her husband. In fact, they haven't even spoken since their breakup four years earlier. The happily-ever-after she remembers in vivid detail--right down to the dress she wore to their wedding--is only one example of what her doctors call a false memory: recollections Lucy's mind made up to fill in the blanks from the coma. Her psychologist explains the condition as honest lying, because while Lucy's memories are false, they still feel incredibly real. Now she has no idea which memories she can trust--a devastating experience not only for Lucy, but also for her family, friends and especially her devoted boyfriend, Matt, whom Lucy remembers merely as a work colleague. When the life Lucy believes she had slams against the reality she's been living for the past four years, she must make a difficult choice about which life she wants to lead, and who she really is."
    # df = np.load('plot_fiction.npy',allow_pickle=True)
    for x in plot:
        m+=1
        # x = tranlate(tranlate([x], "auto2zh"), 'zh2en')
        para.append(x)
        doc2 = en_nlp(x)
        gender_change = change_gender(doc2)
        role,delet =get_role(doc2)
        rolechange = change_rolename(gender_change,role,delet)
        genderwords = change_genderwords(rolechange)
        adjchange,adj,new_adj = find_replace_adj(genderwords, doc2, model,mode='synonym')
        adjchange_new, adj_2, new_adj_2 = find_replace_adj_new(genderwords, doc2, mode='synonym')
        doc3 = en_nlp(genderwords)
        wordnet_sent2 = wordnet_replace(genderwords, doc2,mode='synonym')
        bert_synonyms = wordnet_replace_bert(doc3, mode='synonym')

        adjchange_antonym, adj_antonym, new_adj_antonym = find_replace_adj(genderwords, doc2, model, mode='antonym')
        adjchange_new_antonym, adj_2_antonym, new_adj_2_antonym = find_replace_adj_new(genderwords, doc2,
                                                                                       mode='antonym')
        wordnet_sent2_antonym = wordnet_replace(genderwords, doc2, mode='antonym')
        bert_antonym = wordnet_replace_bert(doc3, mode='antonym')
        trans_w2v_google.append(adjchange_new)
        trans_wordnet.append(wordnet_sent2)
        trans_w2v.append(adjchange)
        trans_bert.append(bert_synonyms)
        trans_wordnet_anto.append(wordnet_sent2_antonym)
        trans_w2v_ant.append(adjchange_antonym)
        trans_w2v_google_ant.append(adjchange_new_antonym)
        trans_bert_ant.append(bert_antonym)
        print('now is  %d'% m)
    return trans_w2v,trans_wordnet,trans_w2v_google,trans_bert,trans_w2v_ant,trans_wordnet_anto,trans_w2v_google_ant,trans_bert_ant

def readtxt(path):
    text=[]
    objFile = open(path,"r",encoding="utf-8")
    while 1:
        line = objFile.readline()
        if not line:
            break
        text.append(line)
    objFile.close()
    return text


def readability(plot_1,plot_2):
    flesch = np.zeros(4)
    flesch_n = np.zeros(4)
    gun = np.zeros(4)
    cole_n = np.zeros(4)
    dale = np.zeros(4)
    gun_n = np.zeros(4)
    cole = np.zeros(4)
    dale_n = np.zeros(4)
    total_score = []
    easy_score = []
    hard_score = []
    easy = []
    diff = []
    for plot1,plot2 in zip(plot_1,plot_2):
        score = 0.
        easyscore = 0.
        hardscore = 0.
        tendtodiff = 0
        tendtoeasy = 0
        flesch1 = textstat.flesch_kincaid_grade(plot1)
        flesch2 = textstat.flesch_kincaid_grade(plot2)
        score_flesh = flesch1-flesch2
        if abs(flesch1-flesch2) <= 1:
            score += abs(score_flesh)
            if flesch1>flesch2: # flesch1 is easier
                flesch[0] +=1
                easyscore +=score_flesh
                tendtoeasy +=1

            else:
                hardscore+=score_flesh
                flesch_n[0]-=1
                tendtodiff +=1

        elif abs(flesch1-flesch2) > 1 and abs(flesch1-flesch2) <= 2:
            score += abs(score_flesh)
            if flesch1>flesch2: # flesch1 is easier
                flesch[1] +=1
                tendtoeasy +=1
                easyscore +=score_flesh

            else:
                tendtodiff +=1
                hardscore+=score_flesh

                flesch_n[1]+=1
        elif abs(flesch1-flesch2) > 2 and abs(flesch1-flesch2) <= 3:
            score += abs(score_flesh)
            if flesch1>flesch2: # flesch1 is easier
                flesch[2] +=1
                tendtoeasy +=1
                easyscore +=score_flesh

            else:
                flesch_n[2]-=1
                tendtodiff +=1
                hardscore+=score_flesh

        elif abs(flesch1-flesch2) > 3:
            score += abs(score_flesh)
            if flesch1>flesch2: # flesch1 is easier
                flesch[3] +=1
                tendtoeasy +=1
                easyscore +=score_flesh

            else:
                flesch_n[3]-=1
                tendtodiff +=1
                hardscore+=score_flesh


        cole1 = textstat.automated_readability_index(plot1)
        cole2 = textstat.automated_readability_index(plot2)
        cole_score = cole1 - cole2
        if abs(cole1 - cole2) <= 1:
            score += abs(cole_score)
            if cole1 > cole2:  #  cole1 is harder
                cole_n[0] += 1
                tendtoeasy +=1
                easyscore +=cole_score

            else:
                tendtodiff +=1
                hardscore+=cole_score

                cole[0] -= 1
        elif abs(cole1 - cole2) > 1 and abs(cole1 - cole2) <= 2:
            score += abs(cole_score)
            if cole1 > cole2:  #  cole1 is harder
                cole_n[1] += 1
                tendtoeasy +=1
                easyscore +=cole_score

            else:
                tendtodiff +=1
                hardscore+=cole_score

                cole[1] -= 1
        elif abs(cole1 - cole2) > 2 and abs(cole1 - cole2) <= 3:
            score += abs(cole_score)
            if cole1 > cole2:  #  cole1 is harder
                cole_n[2] += 1
                tendtoeasy +=1
                easyscore +=cole_score

            else:
                tendtodiff +=1
                hardscore+=cole_score

                cole[2] -= 1
        elif abs(cole1 - cole2) > 3:
            score += abs(cole_score)
            if cole1 > cole2:  #  cole1 is harder
                cole_n[3] += 1
                tendtoeasy +=1
                easyscore +=cole_score

            else:
                cole[3] -= 1
                tendtodiff +=1
                hardscore+=cole_score



        score = score/2
        easyscore = easyscore/2
        hardscore += hardscore/2

        total_score.append(score)
        easy_score.append(easyscore)
        hard_score.append(hardscore)
        easy.append(tendtoeasy)
        diff.append(tendtodiff)
    average = []
    average_n = []

    ave_score = sum(total_score)/len(total_score)
    print('average score = {}'.format(ave_score))
    ave_easy_score = sum(easy_score) / len(easy_score)
    print('average score = {}'.format(ave_easy_score))
    ave_hard_score = sum(hard_score) / len(hard_score)
    print('average score = {}'.format(ave_hard_score))
    easy_score = sum(easy) / (len(easy)+len(diff))
    print('easy percentage = {}'.format(easy_score))
    diff_score = sum(diff) /  (len(easy)+len(diff))
    print('diff percentage = {}'.format(diff_score))
    for i in range(4):
        average.append((flesch[i]+cole[i])/2)
        average_n.append((flesch_n[i]+cole_n[i])/2)
    x = [1,7,]
    x2 = [i+1 for i in x]
    x3 = [i+2 for i in x]
    # x4 = [i+3 for i in x]
    # x5 = [i+4 for i in x]
    tick = [i+2 for i in x]
    bar_width = 0.8
    tick_label = ['same level','differ 1','differ 2','differ more than 2']
    plt.bar(x,flesch,bar_width,align='center',label = 'flesch',alpha = 0.5,color="blue")
    plt.bar(x2,cole,bar_width,align='center',label = 'dale_chall',alpha = 0.5,color="#7EE680")
    plt.bar(x3, average, bar_width, align='center', label='average', alpha=0.5,color="#87CEFA")
    # plt.bar(x4, dale, bar_width, align='center', label='dale_chall', alpha=0.5,color="red")
    # plt.bar(x5, average, bar_width, align='center', label='average', alpha=0.5,color="orange")
    plt.bar(x,flesch_n,bar_width,align='center',color="blue",alpha = 0.5)
    plt.bar(x2,cole,bar_width,align='center',alpha = 0.5,color="#7EE680")
    plt.bar(x3, average, bar_width, align='center', alpha=0.5,color="#87CEFA")
    # plt.bar(x4, dale_n, bar_width, align='center', alpha=0.5,color="red")
    # plt.bar(x5, average_n, bar_width, align='center', alpha=0.5,color="orange")

    plt.xlabel('Differences in the level of education required to understand texts')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks(tick, tick_label)
    plt.title('Differences on Readability between Original and Rewriting Fiction')
    plt.show()


from transformers import AutoModelWithLMHead, AutoTokenizer,AutoModelForCausalLM
# BERT model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model_bert = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")


from gensim.models import KeyedVectors
# GoogleNews model
new_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

en_nlp = st.Pipeline('en')
# our own model + retrofitting with semantic lexicon
model = KeyedVectors.load_word2vec_format('retrofitting\\w2v.txt')
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
test_one(text[0])

