from paddlenlp import Taskflow
from pprint import pprint

parens_path = './项目实战/国网抽取/'
docprompt = Taskflow("document_intelligence", model='docprompt')
pprint(docprompt([{"doc": parens_path+"data/2.png", "prompt": ["日期是多少?"]}]))
pprint(docprompt([{"doc": parens_path+"data/1.png", "prompt": ["额定频率是多少?", "额定电压是多少?",
       "样品型号是什么?", "制造单位是什么?", "制造单位地址在哪里?", "出厂编号是多少?", "委托编号是多少?"]}]))
pprint(docprompt([{"doc": parens_path+"data/3.png", "prompt": ["样品名称是多少?"]}]))
