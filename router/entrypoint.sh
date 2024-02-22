#!/bin/bash

# Check if the first argument is 'serve'
if [ "$1" = "serve" ]; then
    text-embeddings-router
else
    # If the argument is not 'serve', pass all arguments to the original entrypoint
    text-embeddings-router "$@"
fi