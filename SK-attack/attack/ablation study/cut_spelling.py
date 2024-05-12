##仅对重要词中的实词做重要性排序，且添加POS约束保证替换前后词性一致
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
from transformers import logging
from transformers import BertTokenizer,BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification
logging.set_verbosity_error()
import warnings
warnings.filterwarnings('ignore')
from OpenAttack.tags import Tag
from OpenAttack.attackers.classification import Classifier,ClassifierGoal
import OpenAttack
from OpenAttack.utils import check_language
from nltk.corpus import stopwords
import datasets
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
import enchant
from OpenAttack.text_process.tokenizer import PunctTokenizer
from OpenAttack.metric.algorithms.bleu import BLEU
from OpenAttack.text_process.tokenizer import PunctTokenizer
from OpenAttack.metric.algorithms.modification import Modification
from OpenAttack.metric.algorithms.gptlm import GPT2LM

tokenizer = PunctTokenizer()
class SentenceSim(OpenAttack.AttackMetric):##相似度计算
    NAME ="Sentence Similarity"

    def __init__(self):
        self.sim_checker = SentenceTransformer('all-MiniLM-L6-v2')
    def calc_score(self,sen1,sen2):

        emb1, emb2 = self.sim_checker.encode([sen1, sen2], show_progress_bar=False)
        cos_sim = util.pytorch_cos_sim(emb1, emb2)
        return cos_sim.cpu().numpy()[0][0]
    def after_attack(self,input,adversarial_sample):
        return self.calc_score(input["x"], adversarial_sample)
class SKAttacker(OpenAttack.attackers.ClassificationAttacker):  # base.py-classification.py
    @property
    def TAGS(self):## Tags帮助OpenAttack自动检查参数
        return {self.lang_tag, Tag("get_pred", "victim"),Tag("get_prob","Victim")}

    def __init__(self,
                 tokenizer = None,#分词器
                 ):

        if tokenizer is None:
            tokenizer = PunctTokenizer()
        self.tokenizer = tokenizer#分词器
        self.lang_tag = OpenAttack.utils.get_language([self.tokenizer])
        check_language([self.tokenizer], self.lang_tag)#检查语言是否一致

        filter_words = set(stopwords.words('english'))#不包含.../.等标点符号
        self.filter_words = filter_words#使用nltk库的停用词表

        #设置bert预测器
        self.tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
        self.predictor = BertForMaskedLM.from_pretrained('bert-base-uncased')

        #设置检查器：相似度检查+拼写检查
        self.sim_checker = SentenceTransformer('all-MiniLM-L6-v2')
        self.spell_checker = enchant.Dict("en_US")#英文检查器
        #设置关键词提取器
        self.kw_model = KeyBERT(model='paraphrase-MiniLM-L6-v2')

    def attack(self, victim: Classifier, input_, goal: ClassifierGoal):#攻击方法
        #获取原始文本预测标签
        y_orig = victim.get_pred([input_])[0]#获取原句预测标签0/1
        y_adv = 1 - y_orig#对立标签
        orig_probs = victim.get_prob([input_])[0]#预测概率
        orig_prob = orig_probs[y_orig]#原标签预测概率
        orig_prob_adv = orig_probs[y_adv]#对立标签概率
        #使用keybert提取文本关键词
        keywords = self.kw_model.extract_keywords(input_, top_n=10)#提取10个关键词
        #对输入文本进行分句
        new_text = self.remove_empty_lines(input_)
        X = re.split(r"([.!?;])", new_text)
        X.append("")
        X = ["".join(i) for i in zip(X[0::2], X[1::2])]
        #for sentence in input_.split('.'):  # 分句处理，对输入进行处理  (用英文结尾句号.来划分句子)
        #    sentence = sentence.replace('\n', '')  # 去掉句子中的\n换行
        #    if sentence.isspace():# 删除空行
        #        continue
        #    if '?' in sentence:
        #        X.extend(sentence.split('?'))
        #    elif '!' in sentence:
        #        X.extend(sentence.split('!'))
        #    else:
        #        X.append(sentence)
        # 对所有分句计算ISJ重要性得分
        ISJ = {}# 记录X中每个分句的重要性得分
        for idx,sen in enumerate(X):
            X_isj = X[:idx]+X[idx+1:]#删去当前句子
            x_new = self.tokenizer.detokenize(X_isj)#删除后的句子组合成文本
            y_isj = victim.get_pred([x_new])[0]#删除句子后的预测分类
            new_probs = victim.get_prob([x_new])[0]
            new_orig = new_probs[y_orig]
            new_adv = new_probs[y_adv]
            if y_isj == y_orig: #预测相同
                score = orig_prob-new_orig
            else:
                score = orig_prob-new_orig+new_adv-orig_prob_adv
            ISJ[idx]=score#记录句子重要性得分

        Sort_ISJ = sorted(ISJ.items(), key = lambda k:k[1],reverse=True)#按照重要性得分value降序排列，Sort_ISJ是列表
        #根据排序结果将句子分为预干扰Xp和保留Xr
        Xp=[]#预干扰分句
        Xr=[]
        for key in Sort_ISJ:
            if key[1] == 0:#重要性得分ISJ为0
                Xr.append(key)#(句子在X中的索引：句子重要性得分)
            else:
                Xp.append(key)#重要性得分不为0
        #依次对关键句进行干扰，分词后过滤停用词和关键词，然后计算单词重要性得分
        for S in Xp:
            sen_idx = S[0]#句子在X中的索引
            stop = X[sen_idx]#取出索引对应关键句
            stop_orig = X[sen_idx]#未修改的关键句
            stop_orig_prob = victim.get_prob([stop])[0][y_orig]# Y标签下的句子预测概率
            stop_cuts = self.tokenizer.tokenize(stop)# 句子分词，带词性标注
            stop_pos = list(map(lambda x: x[1], stop_cuts))  # 分词词性
            stop_cut = list(map(lambda x: x[0], stop_cuts))  # 分词结果
            #print("origin_stop:",stop_cut)
            stop_filter = [] # 过滤后的词表
            word_score = {} # 单词在原句中的位置：重要性得分
            for idx,word in enumerate(stop_cut):# 过滤停用词、关键词和非实词
                if word.lower() in self.filter_words or word in keywords or stop_pos[idx] == 'other':
                    continue
                stop_filter.append(idx)#过滤后词在stop_cut中的词索引
            #计算单词的重要性得分，Iwk = FY (stop) − FY (stop\wk )
            for idx in stop_filter:#处理原句分词结果
                stop_blank = stop_cut[:idx]+['<UNK>']+stop_cut[idx+1:]#使用"<UNK>"替换单词
                stop_replace = self.tokenizer.detokenize(stop_blank)#组合成句
                score_word = stop_orig_prob-victim.get_prob([stop_replace])[0][y_orig]#得到单词重要性得分,原标签预测值
                word_score[idx]=score_word#该位置上的单词得分
            #单词重要性降序排列
            Sort_L = sorted(word_score.items(), key = lambda k:k[1],reverse=True)#Sort_word是列表
            #对单词依次掩蔽，然后输入预训练的Bert模型进行预测[MASK]
            for instance in Sort_L:#对关键词进行攻击,stop和Xp累计变化
                word_idx = instance[0]#stop_cut中的序号
                word = stop_cut[word_idx]  # 取出对应word
                word_pos = stop_pos[word_idx] #取出单词词性
                bug_sentence = self.selectBUG(word, word_pos, stop,stop_orig, sen_idx, X, victim, goal)#word_idx是不含特殊词汇的原索引，使用三种方式挑选最佳Bug
                stop = bug_sentence#修改stop,累积变化
                X[sen_idx] = stop#替换X中对应句
                x_adv = self.tokenizer.detokenize(X)
                y_new = victim.get_pred([x_adv])[0]
                # 是否攻击成功
                if goal.check(x_adv, y_new):
                    return x_adv

        # Failed
        return None

    def remove_empty_lines(self,text):
        pattern = r"\n\s*\n"  # 匹配连续的空行
        return re.sub(pattern, "\n", text)

    def selectBUG(self, word, word_pos, stop, stop_orig, sen_idx, X, clsf, goal):
        #选择最佳bug，即对模型扰动最大的Bug（逆向分类概率最高）
        bugs = self.generated_bugs(word,word_pos,stop)#生成bug,
        candidate = bugs['replace']+bugs['insert']+bugs['merge']#获取所有候选句
        max_score = float('-inf')  # 计算最佳得分
        best_sentence = stop
        for sentence in candidate:#对每一种变化计算probs
            if self.similarity_check(stop_orig,sentence):#通过相似度检查
                candidate_k = X[:sen_idx]+ [sentence] +X[sen_idx+1:]#对抗样本
                score_k = self.getScore(candidate_k, clsf, goal)#计算该变换的score
                if score_k > max_score:
                    best_sentence = sentence
                    max_score = score_k
        return best_sentence

    def pos_filter(self, ori_pos, new_pos):#词性检查
        if ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['noun', 'verb'])):#相同词性或为动名词互换
            return True
        else:
            return False

    def similarity_check(self,stop_orig,stop_new):
        emb1 = self.sim_checker.encode(stop_orig)#编码计算余弦相似度
        emb2 = self.sim_checker.encode(stop_new)
        cos_sim = util.cos_sim(emb1, emb2).tolist()#tensor转换为list
        if cos_sim[0][0] >= 0.9:
            return True
        else:
            return False


    def getScore(self, candidate, clsf, goal):
        candidate_seg = self.tokenizer.detokenize(candidate)#组合成段
        tempoutput = clsf.get_prob([candidate_seg])[0]#获取模型预测值
        if goal.targeted:
            return tempoutput[goal.target]
        else:
            return - tempoutput[goal.target]#原标签得分最低

    def generated_bugs(self,word,word_pos,stop):
        bugs = {"replace":[],"insert":[],"merge":[]}#替换、插入、合并(候选词列表)
        bugs["replace"] = self.bug_replace(word,word_pos,stop)#替换word
        bugs["insert"] = self.bug_insert(word,stop)#在word后添加MASK
        bugs["merge"] = self.bug_merge(word,stop)#word,word+1设置为一个MASK
        return bugs

    def bug_replace(self,word,word_pos,stop):#在word_idx处替换单词
        #设置MASK标签
        candidate_sentence = []
        try:
            #print("word_replace:",word)
            stop_cuts = self.tokenizer.tokenize(stop,pos_tagging=False)#stop分词
            #print("replace_stop_cuts:",stop_cuts)
            word_idx = stop_cuts.index(word)#获取单词索引位置
            stop_cuts[word_idx] = '[MASK]'#掩蔽word_idx处
            text = self.tokenizer.detokenize(stop_cuts)
            input_ids = self.tokenizer_bert.encode(text, add_special_tokens=True)
            mask_index = input_ids.index(self.tokenizer_bert.mask_token_id)
            input_tensor = torch.tensor([input_ids])
            with torch.no_grad():
                predictions = self.predictor(input_tensor)[0]
            mask_prediction = predictions[0, mask_index].softmax(dim=0)
            top_k = torch.topk(mask_prediction, k=5)
            top_k_tokens = self.tokenizer_bert.convert_ids_to_tokens(top_k.indices.tolist())#得到前五个预测结果
            for idx,token in enumerate(top_k_tokens):
                #if self.spell_checker.check(token) and token not in self.filter_words:#通过拼写检查且预测单词不为原单词
                if token not in self.filter_words:#通过拼写检查且预测单词不为原单词
                    new_list = stop_cuts[:word_idx]+[token]+stop_cuts[word_idx+1:]#替换原词
                    new_sen = self.tokenizer.detokenize(new_list)#组合成句
                    new_pos = list(map(lambda x: x[1], self.tokenizer.tokenize(new_sen)))#带词性分词
                    if self.pos_filter(word_pos,new_pos[word_idx]):#词性检查
                        candidate_sentence.append(new_sen)#候选句
            #print("candidate_replace:",candidate_sentence)
        except:
            pass
        return candidate_sentence


    def bug_insert(self,word,stop):#在word_idx后插入[MASK],不对原句单词进行遮蔽修改
       #插入MASK标签
       candidate_sentence = []
       try:
           #print("word_insert:", word)
           stop_cuts = self.tokenizer.tokenize(stop, pos_tagging=False)  # stop分词
           #print("insert_stop_cuts:", stop_cuts)
           word_idx = stop_cuts.index(word)  # 获取单词索引位置
           stop_copy = stop_cuts[:word_idx+1]+['[MASK]']+stop_cuts[word_idx+1:] # word_idx后插入[MASK]
           text = self.tokenizer.detokenize(stop_copy)
           input_ids = self.tokenizer_bert.encode(text, add_special_tokens=True)
           mask_index = input_ids.index(self.tokenizer_bert.mask_token_id)
           input_tensor = torch.tensor([input_ids])
           with torch.no_grad():
               predictions = self.predictor(input_tensor)[0]
           mask_prediction = predictions[0, mask_index].softmax(dim=0)
           top_k = torch.topk(mask_prediction, k=5)
           top_k_tokens = self.tokenizer_bert.convert_ids_to_tokens(top_k.indices.tolist())  # 得到前五个预测结果
           for token in top_k_tokens:
               #if self.spell_checker.check(token) and token not in self.filter_words:
               if token not in self.filter_words:
                   new_list = stop_cuts[:word_idx+1]+[token]+stop_cuts[word_idx+1:]
                   new_sen = self.tokenizer.detokenize(new_list)
                   new = self.tokenizer.tokenize(new_sen)
                   new_pos = list(map(lambda x: x[1], new))
                   if new_pos[word_idx+1] != 'other':
                       candidate_sentence.append(new_sen)
           #print("candidate_insert:", candidate_sentence)
       except:
           pass
       return candidate_sentence

    def bug_merge(self,word,stop):#合并word_idx和word_idx+1->[MASK]
       #设置MASK
       candidate_sentence = []
       try:
           #print("word_merge:", word)
           stop_cuts = self.tokenizer.tokenize(stop, pos_tagging=False)  # stop分词
           #print("merge_stop_cuts:", stop_cuts)
           word_idx = stop_cuts.index(word)  # 获取单词索引位置
           stop_copy = stop_cuts[:word_idx]+['[MASK]']+stop_cuts[word_idx+2:]  # 掩蔽word_idx,word_idx+1处
           text = self.tokenizer.detokenize(stop_copy)
           input_ids = self.tokenizer_bert.encode(text, add_special_tokens=True)
           mask_index = input_ids.index(self.tokenizer_bert.mask_token_id)
           input_tensor = torch.tensor([input_ids])
           with torch.no_grad():
               predictions = self.predictor(input_tensor)[0]
           mask_prediction = predictions[0, mask_index].softmax(dim=0)
           top_k = torch.topk(mask_prediction, k=5)
           top_k_tokens = self.tokenizer_bert.convert_ids_to_tokens(top_k.indices.tolist())  # 得到前五个预测结果
           for token in top_k_tokens:
               #if self.spell_checker.check(token) and token not in self.filter_words:
               if token not in self.filter_words:
                   new_list = stop_cuts[:word_idx] + [token] + stop_cuts[word_idx+2:]
                   new_sen = self.tokenizer.detokenize(new_list)
                   new = self.tokenizer.tokenize(new_sen)
                   new_pos = list(map(lambda x: x[1], new))
                   if new_pos[word_idx] != 'other':
                       candidate_sentence.append(new_sen)
           #print("candidate_merge:", candidate_sentence)
       except:
           pass
       return candidate_sentence


def dataset_mapping(x):
    return {
        "x": x["text"][:512],#取前512个字符
        "y": 1 if x["ended"] == True else 0,
    }


def main():
    print('Load Victim Model:roberta-base-openai-detector')
    tokenizer1 = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
    model1 = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")
    # print(model)
    victim = OpenAttack.classifiers.TransformersClassifier(model1, tokenizer1, model1.roberta.embeddings.word_embeddings)
    print('New Attacker:SKAttacker')
    attacker = SKAttacker()  # 设计攻击方式
    dataset = datasets.load_dataset('csv', data_files='small-117M-k40.test.csv', split='train[135:150]').map(
        function=dataset_mapping)
    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, victim,metrics=[SentenceSim(),BLEU(PunctTokenizer()),
                                                                                    Modification(PunctTokenizer()), GPT2LM()])  # 开始进行攻击评估
    attack_eval.eval(dataset, visualize=True)


if __name__ == "__main__":
    main()