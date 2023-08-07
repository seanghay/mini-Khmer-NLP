#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import torch

def create_graphemes(text):
  text = text.replace("\u200b", "")
  chunks = []
  for segment in re.finditer(r"([\u1780-\u17df]+)|([\u17e0-\u17e9]+)|(\s+)|([^\u1780-\u17ff\s]+)", text):    
    
    # anything else
    if segment.group(4) is not None:
      chunks.append((segment.group(4), "NS"))

    # whitespace
    if segment.group(3) is not None:
      chunks.append((segment.group(3), "NS"))

    # numbers
    if segment.group(2) is not None:
      chunks.append((segment.group(2), "NS"))

    # khmer characters
    if segment.group(1) is not None:
      for grapheme in re.finditer(r"([\u1780-\u17FF](\u17d2[\u1780-\u17FF]|[\u17B6-\u17D1\u17D3\u17DD])*)", segment.group(1)):
        value = grapheme.group(0)
        type = f"K{len(value)}" if len(value) > 1 else "C"
        chunks.append((value, type))  

  return chunks

def feature_extractor(kccs2int, text):
  g = list(map(lambda x: x[0], create_graphemes(text)))
  return [kccs2int[c] if c in kccs2int else 1 for c in g], g

def tokenize(model, text):
  x, graphemes = feature_extractor(model.kccs2int, text)
  
  with torch.no_grad():
    inputs = torch.tensor(x).unsqueeze(0).long()
    h = model.init_hidden(1)
    val_h = tuple([each.data for each in h])
    pred, _ = model(inputs, val_h)
    pred = torch.sigmoid(pred.cpu())
    pred[pred<0.5] = 0.
    pred[pred>=0.5] = 1.
    pred = pred.int().tolist()
    
    tokens = []
    for i, (grapheme, tag) in enumerate(zip(graphemes, pred)):
      if tag == 1 or i == 0:
        tokens.append(grapheme)
      else:
        tokens[-1] += grapheme
        
    return tokens

if __name__ == "__main__":  
  device = "cpu"
  model = torch.load("word_segmentation_model.pt", map_location=device)
  model.eval()
  tokens = tokenize(model, "ចំណែកជើងទី២ នឹងត្រូវធ្វើឡើងឯប្រទេសកាតា៕")
  print(tokens)
