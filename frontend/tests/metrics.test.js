import test from 'node:test'
import assert from 'node:assert/strict'
import { epochPoint, batchPoint, terminalStatus } from '../src/hooks/metricPoints.js'
test('chart points preserve actual server metrics', () => {
  const epoch = { epoch: 2, train_loss: .12345, val_loss: .98765, train_acc: 71.25, val_acc: 62.5 }
  assert.deepEqual(epochPoint({ ...epoch, type: 'epoch' }), epoch)
  assert.deepEqual(batchPoint({ epoch: 2, batch: 5, loss: .25, running_loss: .75, accuracy: 60 }),
    { label: '2-5', loss: .25, accuracy: 60 })
})
test('terminal events preserve stopped and error state', () => {
  for (const status of ['done','stopped','error']) assert.equal(terminalStatus({ status }), status)
})
