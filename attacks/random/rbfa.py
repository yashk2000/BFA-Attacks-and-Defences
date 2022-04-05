import random
import torch
from models.quantization import quan_Conv2d, quan_Linear, quantize
import operator
from utils.data_conversion import int2bin, bin2int

class BFA(object):
    def __init__(self, criterion, model, k_top=10):

        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        
        # attributes for random attack
        self.module_list = []
        for name, m in model.named_modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                self.module_list.append(name)       

    def flip_bit(self, m):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        if self.k_top is None:
            k_top = m.weight.detach().flatten().__len__()
        else: 
            k_top = self.k_top
        # 1. flatten the gradient tensor to perform topk
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(k_top)
        # update the b_grad to its signed representation
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk]

        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        # convert to twos complement into unsigned integer
        w_bin = int2bin(m.weight.detach().view(-1), m.N_bits).short()
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits,1) & m.b_w.abs().repeat(1,k_top).short()) \
        // m.b_w.abs().repeat(1,k_top).short()
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()

        grads = b_grad_topk.detach().clone()

        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)
        bit2flip = b_grad_topk.clone().view(-1).zero_()

        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            pass

        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
                ^ w_bin_topk

        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        param_flipped = bin2int(w_bin,
                                m.N_bits).view(m.weight.data.size()).float()

        return param_flipped, grads

    def random_flip_one_bit(self, model):
        """
        Note that, the random bit-flip may not support on binary weight quantization.
        """
        chosen_module = random.choice(self.module_list)
        for name, m in model.named_modules():
            if name == chosen_module:
                flatten_weight = m.weight.detach().view(-1)
                chosen_idx = random.choice(range(flatten_weight.__len__()))
                # convert the chosen weight to 2's complement
                bin_w = int2bin(flatten_weight[chosen_idx], m.N_bits).short()
                # randomly select one bit
                bit_idx = random.choice(range(m.N_bits))
                mask = (bin_w.clone().zero_() + 1) * (2**bit_idx)
                bin_w = bin_w ^ mask
                int_w = bin2int(bin_w, m.N_bits).float()
                
                ##############################################
                ###   attack profiling
                ###############################################
                
                weight_mismatch = flatten_weight[chosen_idx] - int_w
                attack_weight_idx = chosen_idx

                print('attacked module:', chosen_module)
                
                attack_log = [] # init an empty list for profile
                
                
                weight_idx = chosen_idx
                weight_prior = flatten_weight[chosen_idx]
                
                wn = weight_prior.item()
                if wn > 0:
                  wn -= 128
                else:
                  wn += 128

                weight_post = torch.tensor(wn, device='cuda')

                print('attacked weight index:', weight_idx)
                print('weight before attack:', weight_prior)
                print('weight after attack:', weight_post)  
                
                tmp_list = ["module_idx", # module index in the net
                            self.bit_counter + 1, # current bit-flip index
                            "loss", # current bit-flip module
                            weight_idx, # attacked weight index in weight tensor
                            weight_prior, # weight magnitude before attack
                            weight_post # weight magnitude after attack
                            ] 
                attack_log.append(tmp_list)                            
                
                self.bit_counter += 1
                #################################
                
                flatten_weight[chosen_idx] = torch.tensor(wn, device='cuda')
                m.weight.data = flatten_weight.view(m.weight.data.size())
                
            
                
        return attack_log
