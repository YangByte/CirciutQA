from ManualProgram import operators
from inspect import getmembers, isfunction
import itertools
import math

constant = [1.5, 3.0, 4.5, 60]
op_dict = {0: 'c_equal',
           1: 'c_double',
           2: 'c_half',
           3: 'c_add',
           4: 'c_minus',
           5: 'c_mul',
           6: 'c_three_mul',
           7: 'c_divide',
           8: 'c_four_mul'
           }

op_list = [op_dict[key] for key in sorted(op_dict.keys())]


class Equations:
    def __init__(self):

        self.op_list = op_list
        self.op_num = {}
        self.call_op = {}
        self.exp_info = None
        self.results = []
        self.max_step = 5
        self.max_len = 7
        for op in self.op_list:
            self.call_op[op] = eval('operators.{}'.format(op))
            self.op_num[op] = self.call_op[op].__code__.co_argcount

    def str2exp(self, inputs):
        inputs = inputs.split(',')
        exp = inputs.copy()
        for i, s in enumerate(inputs):
            if 'n' in s or 'v' in s or 'c' in s:
                exp[i] = s.replace('n', 'N_').replace('v', 'V_').replace('c', 'C_')
            else:
                exp[i] = op_dict[int(s[2:])]
            exp[i] = exp[i].strip()

        self.exp = exp
        return exp

    def excuate_equation(self, exp, source_nums=None):

        if source_nums is None:
            source_nums = self.exp_info['nums']
        vars = []
        idx = 0
        while idx < len(exp):
            op = exp[idx]
            if op not in self.op_list:
                return None
            op_nums = self.op_num[op]
            if idx + op_nums >= len(exp):
                return None
            excuate_nums = []
            for tmp in exp[idx + 1: idx + 1 + op_nums]:
                if tmp[0] == 'N' and int(tmp[-1]) < len(source_nums):
                    excuate_nums.append(source_nums[int(tmp[-1])])
                elif tmp[0] == 'V' and int(tmp[-1]) < len(vars):
                    excuate_nums.append(vars[int(tmp[-1])])
                elif tmp[0] == 'C' and int(tmp[-1]) < len(constant):
                    excuate_nums.append(constant[int(tmp[-1])])
                else:
                    return None
            idx += op_nums + 1
            v = self.call_op[op](*excuate_nums)
            if v is None:
                return None
            vars.append(v)
        return vars


if __name__ == '__main__':
    eq = Equations()
