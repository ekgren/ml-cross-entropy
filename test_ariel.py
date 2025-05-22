import torch
from cut_cross_entropy import linear_cross_entropy # Should pick up your changes

# Example (requires CUDA and appropriate GPU)
if __name__ == '__main__': # Good practice to put test code in a main block
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        device = torch.device("cuda")
        B, S, D, V = 2, 10, 32, 128  # Smaller for quicker testing

        print(f"Using device: {device}")

        # Use bfloat16 as it's common for Ampere
        embeddings = torch.randn(B, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        classifier_weights = torch.randn(V, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(0, V, (B, S), device=device)
        bias = torch.randn(V, device=device, dtype=torch.bfloat16, requires_grad=True)

        print("Inputs created.")

        try:
            loss, max_s = linear_cross_entropy(
                embeddings,
                classifier_weights,
                labels,
                bias=bias,
                reduction="none",
                shift=1 # For next token prediction
            )

            print("\n--- Results ---")
            print("Loss shape:", loss.shape)
            print("Max Softmax Values shape:", max_s.shape)

            # Effective sequence length after shift
            S_effective = S - 1 if S > 1 and 1 > 0 else S

            if loss.numel() > 0 and max_s.numel() > 0: # Check if outputs are not empty
                print("\nLoss (first sequence, first 5 effective tokens):")
                print(loss[0, :min(5, S_effective)])
                print("\nMax Softmax (first sequence, first 5 effective tokens):")
                print(max_s[0, :min(5, S_effective)])
            else:
                print("\nLoss or Max Softmax tensor is empty (e.g., due to shift and sequence length).")


            # Test backward pass
            if loss.requires_grad:
                print("\nTesting backward pass...")
                # Create a scalar loss for backward if reduction was 'none'
                # Handle cases where loss might be empty after shift
                if loss.numel() > 0:
                    scalar_loss_for_backward = loss.mean()
                    scalar_loss_for_backward.backward()
                    print("Gradient for embeddings (sample norm):", embeddings.grad[0,0,:5].norm().item() if embeddings.grad is not None else "N/A")
                    print("Gradient for classifier_weights (sample norm):", classifier_weights.grad[0,:5].norm().item() if classifier_weights.grad is not None else "N/A")
                    if bias.grad is not None:
                        print("Gradient for bias (sample norm):", bias.grad[:5].norm().item())
                    print("Backward pass computation successful.")
                else:
                    print("Loss tensor is empty, skipping backward pass test.")
            else:
                print("\nLoss does not require grad.")

        except Exception as e:
            print(f"\n--- An error occurred ---")
            import traceback
            traceback.print_exc()


    else:
        print("CUDA or bfloat16 not supported on this system. Skipping CCE test.")