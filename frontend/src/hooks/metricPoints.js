// Chart inputs preserve the server metrics; no simulated or smoothed values.
export const epochPoint = data => ({ epoch: data.epoch, train_loss: data.train_loss,
  val_loss: data.val_loss, train_acc: data.train_acc, val_acc: data.val_acc })
export const batchPoint = data => ({ label: `${data.epoch}-${data.batch}`,
  loss: data.loss, accuracy: data.accuracy })
export const terminalStatus = data => data.status || 'done'
