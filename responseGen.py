import os
import csv
import random
import os
import codecs

import math

import torch

from model import *
from util import *

MAX_LENGTH = 10
MIN_COUNT = 3

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


corpus_name = "cornell_movie_dialogs_corpus"
corpus = os.path.join('data', corpus_name)
corpus = os.path.join(os.path.dirname(os.path.realpath(__file__)), corpus)
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

class ResponseGen:
        
    def __init__(self, train=False):
        if(train):
            self.filename = os.path.join(corpus, "movie_lines.txt")
            self.datafile = os.path.join(corpus, "formatted_movie_lines.txt")
    
            self.voc = None
            self.pairs = None

            self.lines = loadLines(self.filename, MOVIE_LINES_FIELDS)
            self.conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"), self.lines, MOVIE_CONVERSATIONS_FIELDS)
            with open(self.datafile, 'w', encoding='utf-8') as outputfile:
                writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
                for pair in extractSentencePairs(self.conversations):
                    writer.writerow(pair)

            save_dir = os.path.join("data", "save")
            self.voc, self.pairs = loadPrepareData(corpus, corpus_name, self.datafile, save_dir)
            
            self.pairs = trimRareWords(self.voc, self.pairs, MIN_COUNT)

            small_batch_size = 5
            batches = batch2TrainData(self.voc, [random.choice(self.pairs) for _ in range(small_batch_size)])
            input_variable, lengths, target_variable, mask, max_target_len = batches           

    
    def train(self):
        model_name = 'cb_model'
        attn_model = 'dot'
        #attn_model = 'general'
        #attn_model = 'concat'
        save_dir = './'
        hidden_size = 500
        encoder_n_layers = 2
        decoder_n_layers = 2
        dropout = 0.1
        batch_size = 64
        loadFilename = None
        checkpoint_iter = 4000
    
        if loadFilename:
            checkpoint = torch.load(loadFilename)
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            self.voc.__dict__ = checkpoint['self.voc_dict']
            

        print('Building encoder and decoder ...')
        embedding = nn.Embedding(self.voc.num_words, hidden_size)
        if loadFilename:
            embedding.load_state_dict(embedding_sd)
        self.encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout)
        self.decoder = LattnDecoder(attn_model, embedding, hidden_size, self.voc.num_words, decoder_n_layers, dropout)
        if loadFilename:
            self.encoder.load_state_dict(encoder_sd)
            self.decoder.load_state_dict(decoder_sd)
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        clip = 50.0

        teacher_forcing_ratio = 1.0
        learning_rate = 0.0001
        decoder_learning_ratio = 5.0
        n_iteration = 4000
        print_every = 200
        save_every = 500
        self.encoder.train()
        self.decoder.train()

        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        if loadFilename:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        print("Starting Training!")
   
        def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                input_variable = input_variable.to(device)
                lengths = lengths.to(device)
                target_variable = target_variable.to(device)
                mask = mask.to(device)
                loss = 0
                print_losses = []
                n_totals = 0
                encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
                decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                decoder_hidden = encoder_hidden[:decoder.n_layers]
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                if use_teacher_forcing:
                    for t in range(max_target_len):
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                        decoder_input = target_variable[t].view(1, -1)
                        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                        loss += mask_loss
                        print_losses.append(mask_loss.item() * nTotal)
                        n_totals += nTotal
                else:
                    for t in range(max_target_len):
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                        _, topi = decoder_output.topk(1)
                        decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                        decoder_input = decoder_input.to(device)
                        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                        loss += mask_loss
                        print_losses.append(mask_loss.item() * nTotal)
                        n_totals += nTotal

                loss.backward()

                _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
                _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

                encoder_optimizer.step()
                decoder_optimizer.step()

                return sum(print_losses) / n_totals
        
        def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

            training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]
            print('Initializing ...')
            start_iteration = 1
            print_loss = 0
            if loadFilename:
                start_iteration = checkpoint['iteration'] + 1

            # Training loop
            print("Training...")
            for iteration in range(start_iteration, n_iteration + 1):
                training_batch = training_batches[iteration - 1]
                input_variable, lengths, target_variable, mask, max_target_len = training_batch
                loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                             decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
                print_loss += loss

                # Print progress
                if iteration % print_every == 0:
                    print_loss_avg = print_loss / print_every
                    print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
                    print_loss = 0

                # Save checkpoint
                if (iteration % save_every == 0):
                    directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torch.save({ 'iteration': iteration, 'en': encoder.state_dict(), 'de': decoder.state_dict(), 'en_opt': encoder_optimizer.state_dict(), 'de_opt': decoder_optimizer.state_dict(), 'loss': loss, 'voc_dict': voc.__dict__, 'embedding': embedding.state_dict()}, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

        
        trainIters(model_name, self.voc, self.pairs, self.encoder, self.decoder, encoder_optimizer, decoder_optimizer,
                  embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                   print_every, save_every, clip, corpus_name, loadFilename)
   
       

    def run(self, inputSeq):
        indexes_batch = [indexesFromSentence(self.voc, inputSeq)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0,1)

        lengths = lengths.to(device)
        input_batch = input_batch.to(device)

        token, scores = greedySearchDecoder(self.encoder, self.decoder)(input_batch, lengths, MAX_LENGTH)
        decode_words = [self.voc.index2word[token.item()] for token in token]
        return decode_words





def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


if(__name__ == "__main__"):
    bot = ResponseGen(True)
    bot.train()

    while(1):
        try:
            input_seq = input(' >')
            output_seq = bot.run(input_seq.lower())
            output_seq[:] = [x for x in output_seq if not (x=='EOS' or x=='PAD')]
            print('Bot:', ' '.join(output_seq))

        except:
            print("Came up with a word not known")