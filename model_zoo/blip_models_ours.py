import os
import torch
import yaml
import subprocess
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from .blip_utils.blip_retrieval import blip_retrieval
from .blip_utils.utils import MetricLogger



class BLIPModelWrapper:
    def __init__(self, device):
        import sys
        sys.path.insert(0, "/home/gamir/DER-Roei/alon/BLIP")
        from argparse import Namespace
        from models.blip_retrieval_vg import blip_retrieval_vg
        import json
        f = open("configs/blip.json")
        blip_config = json.load(f)
        args = Namespace(objects = blip_config["objects"],
                        object_tokens = blip_config["object_tokens"],
                        relations = blip_config["relations"],
                        relation_tokens = blip_config["relation_tokens"],
                        prompt_attention = blip_config["prompt_attention"],
                        prompt_attention_full = blip_config["prompt_attention_full"],
                        mask_layers = blip_config["mask_layers"],
                        text_lora = blip_config["text_lora"],
                        image_lora = blip_config["image_lora"],
                        lora = blip_config["lora"],
                        prompts_lora = blip_config["prompts_lora"],
                        loss_ce = blip_config["loss_ce"],
                        negatives = blip_config["negatives"],
                        random_graph_ablation = blip_config["random_graph_ablation"],
                        vg_loss_lambda = blip_config["vg_loss_lambda"],
                        vg_batch_size = blip_config["vg_batch_size"],
                        device=device
                        )
        model = blip_retrieval_vg(
        pretrained = blip_config["pretrained"],
        med_config = blip_config["med_config"],
        image_size = blip_config["image_size"],
        vit = blip_config["vit"],
        vit_grad_ckpt=blip_config["vit_grad_ckpt"],
        vit_ckpt_layer = blip_config["vit_ckpt_layer"],
        queue_size=blip_config["queue_size"],
        negative_all_rank=blip_config["negative_all_rank"],
        args = args)


        if blip_config["checkpoint"] != "":
            checkpoint = torch.load(blip_config["checkpoint"], map_location='cpu')
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                sd = checkpoint["state_dict"]
                if True and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)



        self.model = model.to(device)
        self.device = device
    
            
        
    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256):
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []  
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i: min(num_text, i+text_bs)]
            text_input = self.model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device) 
            text_output = self.model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
            text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:,0,:]))
            text_embeds.append(text_embed)   
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds,dim=0)
        text_ids = torch.cat(text_ids,dim=0)
        text_atts = torch.cat(text_atts,dim=0)
        text_ids[:,0] = self.model.tokenizer.enc_token_id
        return text_embeds, text_ids, text_atts
    

    @torch.no_grad()
    def get_image_embeddings(self, image_loader):
        image_feats = []
        image_embeds = []
        for batch in tqdm(image_loader):
            image = batch["image"]
            image = image.to(self.device) 
            image_feat = self.model.visual_encoder(image)   
            image_embed = self.model.vision_proj(image_feat[:,0,:])            
            image_embed = F.normalize(image_embed,dim=-1)      

            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)

        image_feats = torch.cat(image_feats,dim=0)
        image_embeds = torch.cat(image_embeds,dim=0)
        return image_feats, image_embeds
        
    
    @torch.no_grad()
    def get_retrieval_scores_dataset(self, loader):
        texts = loader.dataset.text
        metric_logger = MetricLogger(delimiter="  ")
        
        text_embeds, text_ids, text_atts = self.get_text_embeddings(texts)
        image_feats, image_embeds = self.get_image_embeddings(loader)
        sims_matrix = image_embeds @ text_embeds.t()
        score_matrix_i2t = torch.full((image_embeds.shape[0],len(texts)),-100.0).to(self.device)

        num_tasks = 1
        rank = 0
        step = sims_matrix.size(0)//num_tasks + 1
        start = rank*step
        end = min(sims_matrix.size(0),start+step)

        for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, "Evaluation i2T")): 
            topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)

            encoder_output = image_feats[start+i].repeat(self.config['k_test'],1,1).to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(self.device)
            output = self.model.text_encoder(text_ids[topk_idx], 
                                        attention_mask = text_atts[topk_idx],
                                        encoder_hidden_states = encoder_output,
                                        encoder_attention_mask = encoder_att,                             
                                        return_dict = True,
                                       )
            score = self.model.itm_head(output.last_hidden_state[:,0,:])[:,1]
            score_matrix_i2t[start+i,topk_idx] = score + topk_sim
        
        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((len(texts),image_feats.shape[0]),-100.0).to(self.device)

        step = sims_matrix.size(0)//num_tasks + 1
        start = rank*step
        end = min(sims_matrix.size(0),start+step)    

        for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, "Evaluation T2i")): 

            topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)
            encoder_output = image_feats[topk_idx].to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(self.device)
            output = self.model.text_encoder(text_ids[start+i].repeat(self.config['k_test'],1), 
                                        attention_mask = text_atts[start+i].repeat(self.config['k_test'],1),
                                        encoder_hidden_states = encoder_output,
                                        encoder_attention_mask = encoder_att,                             
                                        return_dict = True,
                                       )
            score = self.model.itm_head(output.last_hidden_state[:,0,:])[:,1]
            score_matrix_t2i[start+i,topk_idx] = score + topk_sim

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    
    
    def run_scores_batched(self, image_embeds, image_feats, text_embeds, text_ids, text_atts):
        # Should return something with shape (n_tests, n_image_options, n_text_options)
        # Image embeds and all: (n_tests, n_image_options, embed_dim)
        # Text embeds and all: (n_tests, n_text_options, embed_dim)
        
        # Score matrix should be of the size: (n_tests, n_image_options, n_text_options)
        
        sims_matrix = torch.einsum('ijk,ilk->ijl', image_embeds, text_embeds) # (n_tests, n_image_options, n_text_options)
        
        score_matrix_i2t = torch.full((sims_matrix.shape[0], sims_matrix.shape[1], sims_matrix.shape[2]),-100.0).to(self.device)
        
        for i, sims in enumerate(sims_matrix): 
            for j in range(sims.shape[0]):
                encoder_output = image_feats[i, j].repeat(sims_matrix.shape[2],1,1).to(self.device)
                encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(self.device)
                output = self.model.text_encoder(text_ids[i], 
                                                 attention_mask = text_atts[i],
                                                 encoder_hidden_states = encoder_output,
                                                 encoder_attention_mask = encoder_att,                             
                                                 return_dict = True)
                score = self.model.itm_head(output.last_hidden_state[:,0,:])[:,1]                            
                score_matrix_i2t[i,j] = score + sims[j]
        
        
        sims_matrix = sims_matrix.permute(0,2,1) # (n_tests, n_text_options, n_image_options)
        score_matrix_t2i = torch.full((sims_matrix.shape[0], sims_matrix.shape[1], sims_matrix.shape[2]),-100.0).to(self.device)

        for i, sims in enumerate(sims_matrix): 
            for j in range(sims.shape[0]):
                encoder_output = image_feats[i].to(self.device)
                encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(self.device)
                output = self.model.text_encoder(text_ids[i, j].repeat(sims_matrix.shape[2],1), 
                                                 attention_mask = text_atts[i, j].repeat(sims_matrix.shape[2],1),
                                                 encoder_hidden_states = encoder_output,
                                                 encoder_attention_mask = encoder_att,                             
                                                 return_dict = True)
                score = self.model.itm_head(output.last_hidden_state[:,0,:])[:,1]
                score_matrix_t2i[i,j] = score + sims[j]
        
        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    
        
    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        t2i_scores, i2t_scores = [], []
        for batch in tqdm(joint_loader):
            image_feats = []
            image_embeds = []
            for i_option in batch["image_options"]:        
                image_feat = self.model.visual_encoder(i_option.to(self.device))
                image_embed = self.model.vision_proj(image_feat[:,0,:]) # B x D
                image_embed = F.normalize(image_embed,dim=-1)    
                 
                image_feats.append(image_feat.unsqueeze(1))
                image_embeds.append(image_embed.unsqueeze(1))
            
            image_feats = torch.cat(image_feats,dim=1)
            image_embeds = torch.cat(image_embeds,dim=1)
            
            text_ids = []
            text_embeds = []  
            text_atts = []
            
            for c_option in batch["caption_options"]:
                c_option = list(c_option)
                text_input = self.model.tokenizer(c_option, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device) 
                text_output = self.model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
                text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:,0,:]))
                
                text_embeds.append(text_embed.unsqueeze(1))   
                text_ids.append(text_input.input_ids.unsqueeze(1))
                text_atts.append(text_input.attention_mask.unsqueeze(1))

            text_embeds = torch.cat(text_embeds,dim=1)
            text_ids = torch.cat(text_ids,dim=1)
            text_atts = torch.cat(text_atts,dim=1)
            text_ids[:, :, 0] = self.model.tokenizer.enc_token_id
            
            s_i2t, s_t2i = self.run_scores_batched(image_embeds, image_feats, text_embeds, text_ids, text_atts)
            t2i_scores.append(s_t2i)
            i2t_scores.append(s_i2t)

        t2i_scores = np.concatenate(t2i_scores, axis=0) # N x N_t x N_i
        t2i_scores = np.transpose(t2i_scores, (0, 2, 1)) # N x N_i x N_t
        i2t_scores = np.concatenate(i2t_scores, axis=0) # N x N_i x N_t
        print(t2i_scores.shape, i2t_scores.shape)
        return t2i_scores, i2t_scores