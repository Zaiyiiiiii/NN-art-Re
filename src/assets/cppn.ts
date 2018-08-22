import * as dl from "@tensorflow/tfjs"
import * as nn_art_util from "./nn_art_util"

const MAX_LAYERS = 10

export type ActivationFunction = "tanh" | "sin" | "relu"

const NUM_IMAGE_SPACE_VARIABLES = 3 // x, y, r
const NUM_LATENT_VARIABLES = 2

export class CPPN {
  private inputAtlas: dl.Tensor2D
  private ones: dl.Tensor2D

  private firstLayerWeights: dl.Tensor2D
  private intermediateWeights: dl.Tensor2D[] = []
  private lastLayerWeights: dl.Tensor2D

  private z1Counter = 0
  private z2Counter = 0
  private z1Scale: number
  private z2Scale: number
  private numLayers: number
  private concatAxis = 1

  private selectedActivationFunctionName: ActivationFunction

  private isInferring = false

  constructor(private inferenceCanvas: HTMLCanvasElement) {
    const canvasSize = 128
    this.inferenceCanvas.width = canvasSize
    this.inferenceCanvas.height = canvasSize

    this.inputAtlas = nn_art_util.createInputAtlas(
      canvasSize,
      NUM_IMAGE_SPACE_VARIABLES,
      NUM_LATENT_VARIABLES
    )
    this.ones = dl.ones([this.inputAtlas.shape[0], 1])
  }

  generateWeights(neuronsPerLayer: number, weightsStdev: number) {
    for (let i = 0; i < this.intermediateWeights.length; i++) {
      this.intermediateWeights[i].dispose()
    }
    this.intermediateWeights = []
    if (this.firstLayerWeights != null) {
      this.firstLayerWeights.dispose()
    }
    if (this.lastLayerWeights != null) {
      this.lastLayerWeights.dispose()
    }

    this.firstLayerWeights = dl.truncatedNormal(
      [NUM_IMAGE_SPACE_VARIABLES + NUM_LATENT_VARIABLES, neuronsPerLayer],
      0,
      weightsStdev
    )
    for (let i = 0; i < MAX_LAYERS; i++) {
      this.intermediateWeights.push(
        dl.truncatedNormal([neuronsPerLayer, neuronsPerLayer], 0, weightsStdev)
      )
    }
    this.lastLayerWeights = dl.truncatedNormal(
      [neuronsPerLayer, 3 /** max output channels */],
      0,
      weightsStdev
    )
  }

  setActivationFunction(activationFunction: ActivationFunction) {
    this.selectedActivationFunctionName = activationFunction
  }

  setNumLayers(numLayers: number) {
    this.numLayers = numLayers
  }

  setZ1Scale(z1Scale: number) {
    this.z1Scale = z1Scale
  }

  setZ2Scale(z2Scale: number) {
    this.z2Scale = z2Scale
  }

  start() {
    this.isInferring = true
    this.runInferenceLoop()
  }

  private async runInferenceLoop() {
    if (!this.isInferring) {
      return
    }

    this.z1Counter += 1 / this.z1Scale
    this.z2Counter += 1 / this.z2Scale

    let lastOutput = dl.tidy(() => {
      let z1 = dl.scalar(Math.sin(this.z1Counter))
      let z2 = dl.scalar(Math.cos(this.z2Counter))
      let z1Mat = z1.mul(this.ones) as dl.Tensor2D
      let z2Mat = z2.mul(this.ones) as dl.Tensor2D
      let latentVars = z1Mat.concat(z2Mat, this.concatAxis)

      let activation = (x: dl.Tensor2D) =>
        x[this.selectedActivationFunctionName]()

      let lastOutput = this.inputAtlas.concat(latentVars, this.concatAxis)
      lastOutput = activation(lastOutput.matMul(this.firstLayerWeights))

      for (let i = 0; i < this.numLayers; i++) {
        lastOutput = activation(lastOutput.matMul(this.intermediateWeights[i]))
      }
      z1 = null
      z2 = null
      z1Mat = null
      z2Mat = null
      latentVars = null
      activation = null
      return lastOutput
        .matMul(this.lastLayerWeights)
        .sigmoid()
        .reshape([this.inferenceCanvas.height, this.inferenceCanvas.width, 3])
    })

    await renderToCanvas(lastOutput as dl.Tensor3D, this.inferenceCanvas)
    lastOutput.dispose()
    lastOutput = null
    await dl.nextFrame()
    setTimeout(() => {
      this.runInferenceLoop()
    }, 20)
  }

  stopInferenceLoop() {
    this.isInferring = false
  }
}

// TODO(nsthorat): Move this to a core library util.
async function renderToCanvas(a: dl.Tensor3D, canvas: HTMLCanvasElement) {
  const [height, width] = a.shape
  const ctx = canvas.getContext("2d")
  let imageData = new ImageData(width, height)
  let data = await a.data()
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4
    const k = i * 3
    imageData.data[j + 0] = Math.round(255 * data[k + 0])
    imageData.data[j + 1] = Math.round(255 * data[k + 1])
    imageData.data[j + 2] = Math.round(255 * data[k + 2])
    imageData.data[j + 3] = 255
  }
  data = null
  ctx.putImageData(imageData, 0, 0)
}
