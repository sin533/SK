# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import OpenAttack
import datasets
from OpenAttack.metric.algorithms.bleu import BLEU
from OpenAttack.text_process.tokenizer import PunctTokenizer
from OpenAttack.metric.algorithms.modification import Modification
from OpenAttack.metric.algorithms.gptlm import GPT2LM
from sentence_transformers import util,SentenceTransformer
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
def data_mapping(x):
    return {
        "x": x["text"][:512],
        "y": 1 if x["ended"] == True else 0,
    }

def main():
    print('Load Victim Model:roberta-base-openai-detector')
    tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
    model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")
    #print(model)
    victim = OpenAttack.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)
    print('New Attacker:TextBuggerAttacker')
    attacker = OpenAttack.attackers.TextBuggerAttacker()  # 攻击模型
    dataset = datasets.load_dataset('csv',data_files = 'small-117M-k40.test.csv',split='train[10:40]').map(function=data_mapping)
    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, victim,metrics=[SentenceSim(),BLEU(PunctTokenizer()),
                                                                                    Modification(PunctTokenizer()), GPT2LM()])  # 评估攻击效果
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()