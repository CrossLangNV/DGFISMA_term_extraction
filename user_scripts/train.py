import os
import pickle

from pathlib import Path 

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
       
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, CONFIG_NAME

import plac

from seqeval.metrics import f1_score, accuracy_score

from .utils import SentenceGetter

#helper function
def convert_to_tag( predictions, true_labels, tag_values ):
    
    #sanity check
    assert( len( predictions )==len( true_labels ) )
    
    pred_tags=[]
    true_tags=[]
    
    for pred_sentence, true_sentence in zip( predictions, true_labels):
        pred_sentence_tag=[]
        true_sentence_tag=[]
        for pred_label, true_label in zip( pred_sentence, true_sentence ):
            if tag_values[ true_label ]!="PAD":
                pred_sentence_tag.append(  tag_values[ pred_label ]  )
                true_sentence_tag.append( tag_values[ true_label ]  )
        
        pred_tags.append( pred_sentence_tag )
        true_tags.append( true_sentence_tag )

    return pred_tags, true_tags

@plac.annotations(
    #input-output
    path_training_data=( "Path to the training data (csv file).", ),
    path_output_dir=( "Path to the output folder where trained model will be saved.", ),
    delimiter=( "Delimiter used in training data (csv file).", "option" ),
    #training parameters
    batch_size=( "Batch size used during training.", "option" ),
    seq_length=( "Cut off value for number of tokens in (Bert)tokenized sentence. Bert supports sequences of up to 512 tokens." ),
    epochs=( "Number of training epochs.", "option" ),
    val_size=( "Size of the validation set extracted from the training data.", "option" ),
    seed=( "Seed for train test split.", "option" ),
    gpu=( "GPU id.", "option" )
)
def main( path_training_data:Path,\
          path_output_dir:Path,\
          delimiter:str="˳",\
          batch_size:int=32,\
          seq_length:int=75,\
          epochs:int=10,\
          val_size:float=0.1,\
          seed:int=2020,\
          gpu:int=0):
    '''
    Finetune a BertForTokenClassification model.
    '''
    
    os.makedirs( path_output_dir , exist_ok=True) 

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    device_name=torch.cuda.get_device_name(0) 
        
    print(   'device: {}; numbers of gpus available {}. Name of the device: {}'.format(device , n_gpu , device_name ))
    
    #read the training data
    
    data=pd.read_csv( path_training_data, delimiter=delimiter, engine='python' )
    data=data.fillna( method='ffill'  )
    
    getter=SentenceGetter(data)
    
    tokenized_texts = [[word[0] for word in sentence] for sentence in getter.sentences]

    labels = [[s[2] for s in sentence] for sentence in getter.sentences]

    tag_values = list(set(data["Tag"].values))
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    
    #convert tokenized sentences to ids
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=seq_length, dtype="long", value=0.0,
                              truncating="post", padding="post")

    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=seq_length, value=tag2idx["PAD"], padding="post",
                         dtype="long", truncating="post")

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    #train test split
    
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=seed, test_size=val_size)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=seed, test_size=val_size)

    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)


    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)
    
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )

    model.cuda()

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )

    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))

        pred_tags, valid_tags=convert_to_tag(  predictions, true_labels, tag_values )

        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
        print()

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join( path_output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(path_output_dir, CONFIG_NAME)

        torch.save(model.state_dict(), output_model_file)
        model.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(path_output_dir)

        #during inference you need conversion from ID to BIO tags
        with open( os.path.join( path_output_dir , "tags_vals" ) , "wb" ) as fp:
            pickle.dump( tag_values, fp  )
    