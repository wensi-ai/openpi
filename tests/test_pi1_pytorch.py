"""Standalone unit tests for PI1Pytorch model.

This test can be run independently without pytest or other test frameworks.
Simply run: python tests/test_pi1_pytorch.py
"""

import sys
import os

# Add the src directory to the path so we can import openpi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import MagicMock

import torch

from openpi.models_pytorch.pi1_pytorch import PI1Pytorch


def create_dummy_observation(config, device, batch_size=2):
    """Create a dummy observation object."""
    # Create observation with required attributes
    observation = MagicMock()
    
    # Create dummy images [B, C, H, W] format
    image_shape = (batch_size, 3, 224, 224)
    observation.images = {
        "base_0_rgb": torch.randn(image_shape, device=device),
        "left_wrist_0_rgb": torch.randn(image_shape, device=device),
        "right_wrist_0_rgb": torch.randn(image_shape, device=device),
    }
    
    # Create image masks
    observation.image_masks = {
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
    }
    
    # Create state
    observation.state = torch.randn(batch_size, config.action_dim, device=device)
    
    return observation


def test_forward():
    """Test forward pass of PI1Pytorch."""
    print("\n" + "="*60)
    print("Testing forward() method...")
    print("="*60)
    
    # Create a mock config object
    config = MagicMock()
    config.action_dim = 32
    config.action_horizon = 10
    config.action_expert_variant = "gemma_300m"
    config.pi05 = False
    config.dtype = "bfloat16"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating PI1Pytorch model...")
    model = PI1Pytorch(config)
    model.to(device)
    model.eval()
    
    batch_size = 2
    observation = create_dummy_observation(config, device, batch_size=batch_size)
    
    # Create dummy actions
    actions = torch.randn(
        batch_size, config.action_horizon, config.action_dim, device=device
    )
    
    # Run forward pass
    try:
        print("Running forward pass...")
        loss = model.forward(observation, actions)
        
        # Check output shape
        expected_shape = (batch_size, config.action_horizon, config.action_dim)
        assert loss.shape == expected_shape, f"Loss shape mismatch: got {loss.shape}, expected {expected_shape}"
        
        # Check that loss is non-negative (MSE loss should be >= 0)
        assert torch.all(loss >= 0), "Loss should be non-negative"
        
        print(f"✓ Forward pass successful!")
        print(f"  Loss shape: {loss.shape}")
        print(f"  Mean loss: {loss.mean().item():.4f}")
        print(f"  Min loss: {loss.min().item():.4f}")
        print(f"  Max loss: {loss.max().item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sample_actions():
    """Test sample_actions method of PI1Pytorch."""
    print("\n" + "="*60)
    print("Testing sample_actions() method...")
    print("="*60)
    
    # Create a mock config object
    config = MagicMock()
    config.action_dim = 32
    config.action_horizon = 10
    config.action_expert_variant = "gemma_300m"
    config.pi05 = False
    config.dtype = "bfloat16"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating PI1Pytorch model...")
    model = PI1Pytorch(config)
    model.to(device)
    model.eval()
    
    batch_size = 2
    observation = create_dummy_observation(config, device, batch_size=batch_size)
    
    # Run sample_actions
    try:
        print("Running sample_actions (this may take a moment)...")
        with torch.no_grad():
            actions = model.sample_actions(device, observation, num_steps=5)
        
        # Check output shape
        expected_shape = (batch_size, config.action_horizon, config.action_dim)
        assert actions.shape == expected_shape, f"Actions shape mismatch: got {actions.shape}, expected {expected_shape}"
        
        # Check that actions are finite
        assert torch.all(torch.isfinite(actions)), "Actions should be finite"
        
        print(f"✓ Sample actions successful!")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Mean: {actions.mean().item():.4f}")
        print(f"  Min: {actions.min().item():.4f}")
        print(f"  Max: {actions.max().item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Sample actions failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PI1Pytorch Standalone Test Suite")
    print("="*60)
    
    results = []
    
    # Test forward
    results.append(("forward", test_forward()))
    
    # Test sample_actions
    results.append(("sample_actions", test_sample_actions()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("="*60)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

