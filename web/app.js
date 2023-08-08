import * as ort from 'onnxruntime-web';

async function downloadModelBuffer(update) {
  const response = await fetch("/word_segmentation_model.onnx")
  const reader = response.body.getReader()
  const contentLength = +response.headers.get('Content-Length');
  let receivedLength = 0;
  let chunks = [];
  while(true) {
    const {done, value} = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedLength += value.length;
    update(`${(Math.min(Math.floor(10000 * receivedLength / contentLength) / 100, 100)).toFixed(2)}%`)
  }

  let chunksAll = new Uint8Array(receivedLength); // (4.1)
  let position = 0;
  for(let chunk of chunks) {
    chunksAll.set(chunk, position); // (4.2)
    position += chunk.length;
  }
  return chunksAll
}

function create_graphemes(text) {

  const regex = /([\u1780-\u17df]+)|([\u17e0-\u17e9]+)|(\s+)|([^\u1780-\u17ff\s]+)/gm

  const chunks = [];
  let result;

  while (result = regex.exec(text)) {

    // anything else
    if (result[4]) {
      chunks.push([result[4], 'NS'])
      continue
    }

    // whitespaces
    if (result[3]) {
      chunks.push([result[3], 'NS'])
      continue
    }

    // numbers
    if (result[2]) {
      chunks.push([result[2], 'NS'])
      continue
    }

    // khmer characters
    if (result[1]) {

      const grapheme_regex = /([\u1780-\u17FF](\u17d2[\u1780-\u17FF]|[\u17B6-\u17D1\u17D3\u17DD])*)/gm
      let grapheme_result;

      while (grapheme_result = grapheme_regex.exec(result[1])) {
        const value = grapheme_result[0]
        const type = value.length === 1 ? 'C' : `K${value.length}`
        chunks.push([value, type])
      }

      continue
    }
  }

  return chunks
}

(async function () {
  const $tokens = document.querySelector("#tokens")
  const $textInput = document.querySelector("#text_input")

  const modelBuffer = await downloadModelBuffer(p => {
    $textInput.placeholder = `Downloading model and tokenizer… (${p})`
  });

  const [tokenizers, session] = await Promise.all([
    fetch("/tokenizers.json").then(res => res.json()),
    ort.InferenceSession.create(modelBuffer),
  ])

  $textInput.disabled = false
  $tokens.disabled = false
  $textInput.placeholder = "Input"
  $textInput.value = "ចំណែកជើងទី២ Cambodia Kindom of Wonder នឹងត្រូវធ្វើឡើងឯប្រទេសកាតា៕ Tel: 010123123 🇰🇭"

  $textInput.addEventListener('input', invalidate)
  invalidate()
  async function invalidate() {
    const text = $textInput.value;
    if (!text) {
      $tokens.value = ""
      return
    }
    const chunks = create_graphemes(text)
    const input_ids = chunks.map(item => {
      const id = tokenizers[item[0]]
      if (typeof id !== 'number') return 1
      return id
    })

    const input = new ort.Tensor('int32', input_ids, [1, input_ids.length])
    const { output } = await session.run({ input })
    const prediction = Array.from(output.data);
    const tokens = []
    for (let i = 0; i < chunks.length; i++) {
      if (prediction[i] >= 0.5 || i == 0) {
        tokens.push(chunks[i][0])
      } else {
        tokens[tokens.length - 1] += chunks[i][0]
      }
    }
    $tokens.value = JSON.stringify(tokens)
  }
})()  