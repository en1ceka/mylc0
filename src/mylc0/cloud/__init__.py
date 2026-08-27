"""Cloudflare R2 transport: self-play farm -> object storage -> local trainer.

Nothing in here touches the search, the network or the training targets. It
moves finished chunks and finished networks between machines, and it decides
which generations the trainer may currently sample from.
"""
