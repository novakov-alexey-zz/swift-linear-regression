import Foundation
import _Differentiation

typealias Precision = Float
typealias Dataset = ([Precision], [Precision])

struct Model: Differentiable {
  var weight: Precision
  var bias: Precision

  func callAsFunction(to input: Precision) -> Precision {
    weight * input + bias
  }
}

let random: () -> Precision = { Precision.random(in: 0...1) }

func genDataset(size: Int, for model: Model) -> Dataset {
  var inputs: [Precision] = []
  var outputs: [Precision] = []
  for _ in 0..<size {
    let rnd = random()
    inputs += [rnd]
    outputs += [model(to: rnd)]
  }
  return (inputs, outputs)
}

func mse(_ y: Precision, _ yHat: Precision) -> Precision {
  pow(y - yHat, 2)
}

extension Array {
  func group(_ size: Int) -> [[Element]] {
    stride(from: 0, to: count, by: size).map {
      Array(self[$0..<Swift.min($0 + size, count)])
    }
  }

  func split(_ ratio: Double) -> (left: [Element], right: [Element]) {
    let count = Double(self.count) * ratio
    let countOrZero = (count < 1) ? 0 : Int(count)
    let pos = self.count - countOrZero
    let leftSplit = self[0..<pos]
    let rightSplit = self[pos..<self.count]
    return (left: Array(leftSplit), right: Array(rightSplit))
  }
}

extension Array where Element: FloatingPoint {
  func mean() -> Element {
    self.reduce(0, +) / Element(self.count)
  }
}

let realModel = Model(weight: random(), bias: random())
let (xDataset, yDataset) = genDataset(size: 10000, for: realModel)
let ratio = 0.2
let (xTrain, xTest) = xDataset.split(ratio)
let (yTrain, yTest) = yDataset.split(ratio)

let epochs = 40
let learningRate: Precision = 0.001
let batchSize = 16
var model = Model(weight: random(), bias: 0)

for epoch in 1...epochs {
  var losses: [Precision] = []

  for (xBatch, yBatch) in zip(xTrain.group(batchSize), yTrain.group(batchSize)) {

    let (loss, 𝛁batch) =
      zip(xBatch, yBatch)
      .reduce(
        (Precision(0), Model.TangentVector.init(weight: Precision(0), bias: Precision(0)))
      ) { (acc, data) in
        let (x, y) = data
        let (loss, 𝛁model) = valueWithGradient(at: model) { model -> Precision in
          let yHat = model(to: x)
          return mse(y, yHat)
        }
        return (acc.0 + loss, acc.1 + 𝛁model)
      }

    losses += [loss / Precision(xBatch.count)]
    let 𝛁w = 𝛁batch.weight / Precision(xBatch.count)
    let 𝛁b = 𝛁batch.bias / Precision(xBatch.count)
    model = Model(
      weight: model.weight - learningRate * 𝛁w, bias: model.bias - learningRate * 𝛁b)
  }
  print("epoch: \(epoch), loss: \(losses.mean())")
}

print("real model: \(realModel)")
print("trained model: \(model)")
