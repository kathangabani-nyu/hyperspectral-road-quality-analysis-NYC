"""
Interactive Manual Masking Tool for Pavement Annotation

Allows user to manually mark pavement areas by clicking/drawing on the RGB image.
The mask can then be used to improve classification or as ground truth.

Usage:
    python manual_mask_tool.py
"""

# CRITICAL: Set backend BEFORE any matplotlib imports
import matplotlib
matplotlib.use('TkAgg', force=True)  # Force interactive backend

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import rasterio
import os
import advanced_config as config
from data_loader import load_hyperspectral_data

# Import visualizer functions we need (but avoid its backend setting)
from visualizer import percentile_stretch


class ManualMaskTool:
    def __init__(self, rgb_image, existing_mask=None, save_path=None):
        """
        Initialize manual masking tool.
        
        Parameters:
        -----------
        rgb_image : ndarray
            RGB composite image for visualization
        existing_mask : ndarray, optional
            Existing binary mask to edit
        save_path : str, optional
            Path to save the mask
        """
        self.rgb_image = rgb_image
        self.height, self.width = rgb_image.shape[:2]
        
        if existing_mask is not None:
            self.mask = existing_mask.copy().astype(bool)
        else:
            self.mask = np.zeros((self.height, self.width), dtype=bool)
        
        self.save_path = save_path or os.path.join(config.OUTPUT_DIR, "manual_pavement_mask.npy")
        
        # Drawing state
        self.drawing = False
        self.mode = 'add'  # 'add' or 'remove'
        self.brush_size = 5
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(15, 12))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        # Display
        self.update_display()
        
        # Add controls
        self.add_controls()
        
    def add_controls(self):
        """Add control buttons and widgets."""
        # Mode selector
        ax_mode = plt.axes([0.02, 0.02, 0.15, 0.1])
        self.radio = RadioButtons(ax_mode, ('Add', 'Remove'))
        self.radio.on_clicked(self.set_mode)
        
        # Brush size
        ax_brush = plt.axes([0.02, 0.15, 0.15, 0.05])
        self.brush_text = ax_brush.text(0.5, 0.5, f'Brush: {self.brush_size}', 
                                        ha='center', va='center', fontsize=12)
        ax_brush.set_xticks([])
        ax_brush.set_yticks([])
        
        # Buttons
        ax_save = plt.axes([0.85, 0.02, 0.1, 0.05])
        self.btn_save = Button(ax_save, 'Save Mask')
        self.btn_save.on_clicked(self.save_mask)
        
        ax_clear = plt.axes([0.85, 0.08, 0.1, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_clear.on_clicked(self.clear_mask)
        
        ax_brush_up = plt.axes([0.20, 0.15, 0.05, 0.05])
        self.btn_brush_up = Button(ax_brush_up, '+')
        self.btn_brush_up.on_clicked(self.brush_up)
        
        ax_brush_down = plt.axes([0.02, 0.15, 0.05, 0.05])
        self.btn_brush_down = Button(ax_brush_down, '-')
        self.btn_brush_down.on_clicked(self.brush_down)
        
        # Instructions
        instructions = (
            "Instructions:\n"
            "1. Click 'Add' to mark pavement areas\n"
            "2. Click 'Remove' to unmark areas\n"
            "3. Click and drag to draw\n"
            "4. Use +/- to adjust brush size\n"
            "5. Click 'Save Mask' when done"
        )
        ax_info = plt.axes([0.02, 0.25, 0.2, 0.15])
        ax_info.text(0.05, 0.5, instructions, fontsize=9, va='center',
                    family='monospace', transform=ax_info.transAxes)
        ax_info.set_xticks([])
        ax_info.set_yticks([])
    
    def set_mode(self, label):
        """Set drawing mode."""
        self.mode = 'add' if label == 'Add' else 'remove'
    
    def brush_up(self, event):
        """Increase brush size."""
        self.brush_size = min(50, self.brush_size + 2)
        self.brush_text.set_text(f'Brush: {self.brush_size}')
        self.fig.canvas.draw()
    
    def brush_down(self, event):
        """Decrease brush size."""
        self.brush_size = max(1, self.brush_size - 2)
        self.brush_text.set_text(f'Brush: {self.brush_size}')
        self.fig.canvas.draw()
    
    def draw_circle(self, x, y):
        """Draw a circle at (x, y) with current brush size."""
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        dist_sq = (x_coords - x)**2 + (y_coords - y)**2
        circle_mask = dist_sq <= (self.brush_size ** 2)
        
        if self.mode == 'add':
            self.mask[circle_mask] = True
        else:
            self.mask[circle_mask] = False
    
    def on_click(self, event):
        """Handle mouse click."""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left click
            self.drawing = True
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.width and 0 <= y < self.height:
                self.draw_circle(x, y)
                self.update_display()
    
    def on_motion(self, event):
        """Handle mouse motion while drawing."""
        if not self.drawing or event.inaxes != self.ax:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < self.width and 0 <= y < self.height:
            self.draw_circle(x, y)
            self.update_display()
    
    def on_release(self, event):
        """Handle mouse release."""
        self.drawing = False
    
    def update_display(self):
        """Update the display with current mask overlay."""
        self.ax.clear()
        
        # Show RGB image
        self.ax.imshow(self.rgb_image)
        
        # Overlay mask in semi-transparent red
        overlay = np.zeros((self.height, self.width, 4))
        overlay[:, :, 0] = 1.0  # Red
        overlay[:, :, 3] = self.mask.astype(float) * 0.5  # 50% opacity
        
        self.ax.imshow(overlay)
        self.ax.set_title(f'Manual Pavement Masking (Mode: {self.mode.capitalize()}, '
                         f'Brush: {self.brush_size}, '
                         f'Pavement pixels: {self.mask.sum()})',
                         fontsize=14, fontweight='bold')
        self.ax.axis('off')
        
        self.fig.canvas.draw()
    
    def clear_mask(self, event):
        """Clear the entire mask."""
        self.mask.fill(False)
        self.update_display()
    
    def save_mask(self, event):
        """Save the mask to file."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Save as numpy array
        np.save(self.save_path, self.mask.astype(int))
        print(f"\nMask saved to: {self.save_path}")
        print(f"Pavement pixels: {self.mask.sum()} ({self.mask.sum() / self.mask.size * 100:.2f}%)")
        
        # Also save as visualization
        vis_path = self.save_path.replace('.npy', '_visualization.png')
        self.fig.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {vis_path}")
        
        plt.close(self.fig)


def load_existing_mask(mask_path):
    """Load existing mask if it exists."""
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        return mask.astype(bool)
    return None


def main():
    """Run manual masking tool."""
    print("="*70)
    print("MANUAL PAVEMENT MASKING TOOL")
    print("="*70)
    
    # Load data - only RGB bands needed for visualization (much faster!)
    print("\nLoading hyperspectral data (RGB bands only for faster loading)...")
    
    # Only load the 3 bands needed for RGB visualization (1-indexed for rasterio)
    rgb_band_indices = [
        config.RED_BAND_IDX + 1,    # Convert 0-indexed to 1-indexed
        config.GREEN_BAND_IDX + 1,
        config.BLUE_BAND_IDX + 1
    ]
    
    # Load only RGB bands
    from rasterio.windows import Window
    import rasterio
    
    with rasterio.open(config.HYPERSPECTRAL_PATH) as src:
        # Get window from config
        start_x = getattr(config, 'START_X', 0)
        start_y = getattr(config, 'START_Y', 0)
        test_size_x = getattr(config, 'TEST_SIZE_X', 800)
        test_size_y = getattr(config, 'TEST_SIZE_Y', 800)
        
        window = Window(start_x, start_y, test_size_x, test_size_y)
        
        print(f"Reading RGB bands {rgb_band_indices} from window {test_size_x}x{test_size_y}...")
        # Read only the 3 RGB bands (much faster than all 288!)
        rgb_data = src.read(rgb_band_indices, window=window)  # Shape: (3, height, width)
        
        height, width = rgb_data.shape[1], rgb_data.shape[2]
        print(f"Loaded: {height}x{width} pixels")
    
    # Create RGB composite directly from the 3 bands
    print("\nCreating RGB composite...")
    rgb_composite = np.zeros((height, width, 3), dtype=np.float32)
    rgb_composite[:, :, 0] = rgb_data[0, :, :].astype(np.float32)  # Red
    rgb_composite[:, :, 1] = rgb_data[1, :, :].astype(np.float32)   # Green
    rgb_composite[:, :, 2] = rgb_data[2, :, :].astype(np.float32)   # Blue
    
    # Apply percentile stretching
    from visualizer import percentile_stretch
    for i in range(3):
        rgb_composite[:, :, i] = percentile_stretch(
            rgb_composite[:, :, i],
            config.RGB_PERCENTILE_LOW,
            config.RGB_PERCENTILE_HIGH
        )
    
    print(f"RGB composite ready: {rgb_composite.shape}")
    
    # Check for existing mask
    mask_path = os.path.join(config.OUTPUT_DIR, "manual_pavement_mask.npy")
    existing_mask = load_existing_mask(mask_path)
    
    if existing_mask is not None:
        print(f"\nFound existing mask: {mask_path}")
        print(f"  Pavement pixels: {existing_mask.sum()} ({existing_mask.sum() / existing_mask.size * 100:.2f}%)")
        use_existing = input("Load existing mask? (y/n): ").lower().strip() == 'y'
        if not use_existing:
            existing_mask = None
    
    # Launch tool
    print("\nLaunching interactive masking tool...")
    print("Close the window when done (or click 'Save Mask' button).")
    
    tool = ManualMaskTool(rgb_composite, existing_mask, mask_path)
    plt.show()
    
    print("\nManual masking complete!")


if __name__ == "__main__":
    main()

