export function hasPendingRagFiles(files) {
  return files.some((file) => file.status === 'queued' || file.status === 'processing');
}
