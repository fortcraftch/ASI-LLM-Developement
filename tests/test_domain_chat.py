import unittest
import tiktoken
import torch
from types import SimpleNamespace
from asi.runtime.generation import domain_chat_context,generate


class DomainChatTest(unittest.TestCase):
    def test_stop_reasons(self):
        class Model:
            config=SimpleNamespace(block_size=6)
            def __init__(self,chosen):self.chosen=chosen
            def __call__(self,ids,start_pos=0):
                values=torch.full((*ids.shape,4),-20.)
                values[:,:,self.chosen]=20.
                return values,None
        enc=SimpleNamespace(encode=lambda *a,**k:[1,2],decode=lambda ids:','.join(map(str,ids)),eot_token=3,n_vocab=4)
        eos=generate(Model(3),enc,'prompt','cpu',3,1.,1,return_info=True)
        self.assertEqual(eos['stop_reason'],'eos');self.assertEqual(eos['generated_tokens'],1)
        limit=generate(Model(0),enc,'prompt','cpu',2,1.,1,return_info=True)
        self.assertEqual(limit['stop_reason'],'max_new_tokens');self.assertEqual(limit['generated_tokens'],2)
        ctx=generate(Model(0),enc,'prompt','cpu',9,1.,1,return_info=True,incremental=False)
        self.assertEqual(ctx['stop_reason'],'context_limit');self.assertEqual(ctx['generated_tokens'],4)
    def test_oasst_history_and_user_only_routing(self):
        enc=tiktoken.get_encoding('gpt2')
        context,route=domain_chat_context(enc,[('user','Hi'),('assistant','Hello')],'Explain Python','oasst',128)
        self.assertEqual(context,'<|user|>\nHi\n<|assistant|>\nHello<|endoftext|><|user|>\nExplain Python\n<|assistant|>\n')
        self.assertEqual(route,'Hi\nExplain Python')
        self.assertNotIn('Hello',route)

    def test_truncation_reserves_answer_space(self):
        enc=tiktoken.get_encoding('gpt2')
        context,_=domain_chat_context(enc,[('user','old '*100),('assistant','answer '*100)],'new question','oasst',30)
        self.assertLessEqual(len(enc.encode(context,allowed_special={'<|endoftext|>'})),30)
        self.assertTrue(context.endswith('new question\n<|assistant|>\n'))
        self.assertEqual(domain_chat_context(enc,[],'Hello','completion',30),('Hello','Hello'))


if __name__=='__main__':unittest.main()
